!pip install soundfile -q

# ============================================================================
# OPTIMIZED MAMBA SPEAKER VERIFICATION - RESEARCH-BACKED
# Architecture: Wav2Vec2 → Bidirectional Mamba (FIXED) → Attention Pooling
# Improvements: Stable SSM, better training, attention pooling, warmup
# ============================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import random
import os
import gc
import math
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("GPU SETUP")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.cuda.empty_cache()
    print("✓ GPU cache cleared")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================================
# DATA LOADING
# ============================================================================

print("\n" + "="*70)
print("LOADING AND PREPARING DATA")
print("="*70)

df = pd.read_csv('/content/drive/MyDrive/VoxCeleb1_WebDataset/vox1_processed_dataset.csv')

if 'utterance' in df.columns:
    df = df.drop('utterance', axis=1)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✓ Dataset loaded and shuffled")
print(f"Total samples: {len(df)}")
print(f"Unique speakers: {df['speaker_id'].nunique()}")

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['speaker_id'])
num_speakers = df['label'].nunique()

print(f"\nNumber of speaker classes: {num_speakers}")

train_df, val_df = train_test_split(
    df,
    test_size=0.14,
    stratify=df['label'],
    random_state=42
)

print(f"\n✓ Data split complete:")
print(f"  Train samples: {len(train_df)}")
print(f"  Validation samples: {len(val_df)}")

# ============================================================================
# IMPROVED AUDIO AUGMENTATION
# ============================================================================

class AudioAugmentation:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def add_noise(self, waveform, noise_level=0.003):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise

    def time_stretch(self, waveform, rate=None):
        if rate is None:
            rate = random.choice([0.95, 1.0, 1.05])

        if rate == 1.0:
            return waveform

        orig_length = waveform.shape[0]
        new_length = int(orig_length / rate)

        if new_length != orig_length:
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0).unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=False
            ).squeeze()

        return waveform

    def augment(self, waveform, augment_prob=0.3):
        if random.random() < augment_prob:
            if random.random() < 0.5:
                waveform = self.add_noise(waveform)
            if random.random() < 0.3:
                waveform = self.time_stretch(waveform)

        return waveform

# ============================================================================
# OPTIMIZED MAMBA ARCHITECTURE
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.weight

class OptimizedSelectiveSSM(nn.Module):
    """
    OPTIMIZED Selective State Space Model
    - Better numerical stability
    - Proper initialization
    - Gradient-friendly implementation
    """
    def __init__(self, d_model: int, d_state: int, dt_rank: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Time step projection
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # Initialize A with better stability (log-space initialization)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection parameter
        self.D = nn.Parameter(torch.ones(d_model))

        # B and C projections
        self.BC_proj = nn.Linear(d_model, d_state * 2, bias=False)
        self.delta_proj = nn.Linear(d_model, dt_rank, bias=True)

        # Improved initialization for stability
        nn.init.uniform_(self.dt_proj.bias, 0.001, 0.1)
        nn.init.uniform_(self.delta_proj.bias, -0.05, 0.05)
        nn.init.normal_(self.dt_proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Compute delta (time steps) with stability
        delta = self.delta_proj(x)
        delta = F.softplus(self.dt_proj(delta))
        delta = torch.clamp(delta, min=1e-4, max=1.0)  # Prevent extreme values

        # Compute B and C
        BC = self.BC_proj(x)
        B, C = BC.chunk(2, dim=-1)

        # Compute A (transition matrix)
        A = -torch.exp(self.A_log.clamp(max=10.0))  # Prevent overflow

        # SSM recurrence with improved stability
        y = torch.zeros_like(x)
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            delta_t = delta[:, t, :]

            # Discretize A and B with clamping
            A_disc = torch.exp(delta_t.unsqueeze(-1) * A)
            A_disc = torch.clamp(A_disc, max=1.0)  # Numerical stability

            B_disc = delta_t.unsqueeze(-1) * B[:, t, :].unsqueeze(1)

            # State update
            x_t = x[:, t, :].unsqueeze(-1)
            h = A_disc * h + B_disc * x_t

            # Output computation
            y_t = (C[:, t, :].unsqueeze(1) @ h.transpose(1, 2)).squeeze(1)
            y_t = y_t + self.D * x[:, t, :]
            y[:, t, :] = y_t

        return y

class OptimizedMambaBlock(nn.Module):
    """Optimized Mamba block with better stability"""
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None, expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.d_conv = d_conv

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        self.ssm = OptimizedSelectiveSSM(self.d_inner, d_state, self.dt_rank)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.silu = nn.SiLU()

        # Better initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.norm(x)
        proj = self.in_proj(x)
        x_path, z_gate = proj.chunk(2, dim=-1)

        # Causal convolution
        x_path = x_path.transpose(1, 2)
        x_path = self.conv1d(x_path)[:, :, :x_path.shape[-1]]
        x_path = x_path.transpose(1, 2)
        x_path = self.silu(x_path)

        # SSM
        y_ssm = self.ssm(x_path)

        # Gating
        z_gate = self.silu(z_gate)
        out = y_ssm * z_gate
        out = self.out_proj(out)

        # Residual connection
        out = out + skip

        return out

class BidirectionalMamba(nn.Module):
    """Bidirectional Mamba for capturing both directions"""
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None, expand: int = 2):
        super().__init__()
        self.forward_mamba = OptimizedMambaBlock(d_model, d_state, dt_rank, expand)
        self.backward_mamba = OptimizedMambaBlock(d_model, d_state, dt_rank, expand)
        self.merge = nn.Linear(d_model * 2, d_model, bias=False)
        self.norm = RMSNorm(d_model * 2)

        # Better initialization
        nn.init.xavier_uniform_(self.merge.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self.forward_mamba(x)
        bwd = self.backward_mamba(torch.flip(x, dims=[1]))
        bwd = torch.flip(bwd, dims=[1])

        merged = torch.cat([fwd, bwd], dim=-1)
        merged = self.norm(merged)
        output = self.merge(merged)

        return output

class AttentionPooling(nn.Module):
    """Learnable attention-based pooling - better than mean pooling"""
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        # Initialize properly
        for module in self.attention.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        attn_weights = self.attention(x)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted = (x * attn_weights).sum(dim=1)  # [batch, d_model]
        return weighted

class OptimizedMambaSpeakerClassifier(nn.Module):
    """
    Optimized Mamba-based Speaker Classifier
    - Bidirectional Mamba blocks (research-backed for audio)
    - Attention pooling (better than mean)
    - Proper initialization for stability
    """
    def __init__(
        self,
        num_speakers: int,
        wav2vec_model_name: str = "facebook/wav2vec2-large-xlsr-53",
        d_state: int = 16,
        expand: int = 2,
        num_mamba_blocks: int = 3,
        unfreeze_last_n_layers: int = 3
    ):
        super().__init__()

        # Load Wav2Vec2
        self.wav2vec_encoder = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        self.d_model = self.wav2vec_encoder.config.hidden_size

        # Partial freezing
        for param in self.wav2vec_encoder.feature_extractor.parameters():
            param.requires_grad = False

        for param in self.wav2vec_encoder.encoder.parameters():
            param.requires_grad = False

        total_layers = len(self.wav2vec_encoder.encoder.layers)
        for i in range(total_layers - unfreeze_last_n_layers, total_layers):
            for param in self.wav2vec_encoder.encoder.layers[i].parameters():
                param.requires_grad = True

        print(f"✓ Wav2Vec2: {total_layers} layers, trainable: last {unfreeze_last_n_layers}")

        # Bidirectional Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            BidirectionalMamba(self.d_model, d_state, self.d_model // 16, expand)
            for _ in range(num_mamba_blocks)
        ])

        print(f"✓ Bidirectional Mamba: {num_mamba_blocks} blocks (OPTIMIZED SSM)")

        # Attention pooling instead of mean pooling
        self.attention_pool = AttentionPooling(self.d_model)

        # Classifier
        self.norm = RMSNorm(self.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_speakers)
        )

        # Initialize classifier
        self._init_classifier()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal params: {total_params:,}, Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    def _init_classifier(self):
        """Proper weight initialization for classifier"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_values: torch.Tensor, return_embeddings: bool = False):
        # Extract Wav2Vec2 features
        outputs = self.wav2vec_encoder(input_values)
        x = outputs.last_hidden_state

        # Mamba encoding
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)

        # Normalize
        x = self.norm(x)

        # Attention pooling
        pooled = self.attention_pool(x)

        if return_embeddings:
            return pooled

        # Classification
        logits = self.classifier(pooled)
        return logits, pooled

# ============================================================================
# DATASET
# ============================================================================

class SpeakerDataset(Dataset):
    def __init__(self, dataframe, feature_extractor, max_duration=4.0, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.max_length = int(max_duration * 16000)
        self.augment = augment
        if self.augment:
            self.augmentor = AudioAugmentation(16000)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            waveform, sr = torchaudio.load(row['audio_path'])
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)

            waveform = waveform.squeeze()

            if self.augment:
                waveform = self.augmentor.augment(waveform, augment_prob=0.3)

            waveform = waveform.numpy()

            if len(waveform) > self.max_length:
                if self.augment:
                    start = random.randint(0, len(waveform) - self.max_length)
                else:
                    start = (len(waveform) - self.max_length) // 2
                waveform = waveform[start:start + self.max_length]
            else:
                waveform = np.pad(waveform, (0, self.max_length - len(waveform)))

            inputs = self.feature_extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=False
            )

            return inputs.input_values.squeeze(0), row['label']
        except Exception as e:
            dummy = np.zeros(self.max_length)
            inputs = self.feature_extractor(dummy, sampling_rate=16000, return_tensors="pt", padding=False)
            return inputs.input_values.squeeze(0), row['label']

def collate_fn(batch):
    input_values, labels = zip(*batch)
    return torch.stack(input_values), torch.tensor(labels)

# ============================================================================
# IMPROVED LEARNING RATE SCHEDULE
# ============================================================================

class WarmupCosineScheduler:
    """Warmup + Cosine annealing - better than step decay"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_scale', 1.0)

        self.current_epoch += 1
        return lr

# ============================================================================
# CLEANUP
# ============================================================================

vars_to_delete = ['model', 'optimizer', 'scheduler', 'feature_extractor',
                  'train_loader', 'val_loader', 'train_dataset', 'val_dataset']

for var in vars_to_delete:
    if var in globals():
        del globals()[var]

gc.collect()
torch.cuda.empty_cache()

print(f"✓ GPU Memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB used")

# ============================================================================
# INITIALIZE
# ============================================================================

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

train_dataset = SpeakerDataset(train_df, feature_extractor, max_duration=4.0, augment=True)
val_dataset = SpeakerDataset(val_df, feature_extractor, max_duration=4.0, augment=False)

# Memory-optimized batch sizes
train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn,
    drop_last=True,
    prefetch_factor=2,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=28,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn,
    drop_last=False,
    prefetch_factor=2,
    persistent_workers=True
)

print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Initialize optimized Mamba model
model = OptimizedMambaSpeakerClassifier(
    num_speakers=num_speakers,
    d_state=16,              # Standard size for stability
    expand=2,                # Standard expansion
    num_mamba_blocks=3,      # Good depth
    unfreeze_last_n_layers=3
).to(device)

print("✓ Model initialized - OPTIMIZED MAMBA!")
torch.cuda.empty_cache()

# ============================================================================
# OPTIMIZER
# ============================================================================

def get_optimizer(model, base_lr=5e-4, wav2vec_lr_scale=0.1, weight_decay=0.01):
    wav2vec_params = []
    for n, p in model.wav2vec_encoder.named_parameters():
        if p.requires_grad:
            wav2vec_params.append(p)

    other_params = []
    for n, p in model.named_parameters():
        if p.requires_grad and 'wav2vec_encoder' not in n:
            other_params.append(p)

    optimizer = AdamW([
        {'params': wav2vec_params, 'lr': base_lr * wav2vec_lr_scale, 'lr_scale': wav2vec_lr_scale},
        {'params': other_params, 'lr': base_lr, 'lr_scale': 1.0},
    ], weight_decay=weight_decay)

    print(f"✓ Optimizer: Wav2Vec2 LR={base_lr*wav2vec_lr_scale:.2e}, Mamba/Classifier LR={base_lr:.2e}")
    return optimizer

base_lr = 5e-4
optimizer = get_optimizer(model, base_lr=base_lr, wav2vec_lr_scale=0.1)
criterion = CrossEntropyLoss(label_smoothing=0.05)

epochs = 15
warmup_epochs = 2
scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, base_lr)

print(f"✓ Scheduler: {warmup_epochs} warmup epochs, {epochs} total epochs")

# ============================================================================
# CHECKPOINT
# ============================================================================

checkpoint_path = "/content/drive/MyDrive/VoxCeleb1_WebDataset/models/optimized_mamba_best.pth"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

start_epoch = 0
best_val_acc = 0.0

if os.path.exists(checkpoint_path):
    print(f"\n✓ Found checkpoint - resuming training")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['val_acc']
    scheduler.current_epoch = start_epoch
    print(f"  Resuming from epoch {start_epoch}/{epochs}")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    del checkpoint
    torch.cuda.empty_cache()
else:
    print(f"\n✓ Starting fresh training")

print(f"\n{'='*70}")
print("STARTING TRAINING - OPTIMIZED MAMBA (RESEARCH-BACKED)")
print(f"{'='*70}\n")

# ============================================================================
# TRAINING LOOP
# ============================================================================

for epoch in range(start_epoch, epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    current_lr = scheduler.step()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train] LR={current_lr:.2e}", leave=False, dynamic_ncols=True)

    for batch_idx, (batch_input, batch_labels) in enumerate(pbar):
        batch_input = batch_input.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits, _ = model(batch_input)
        loss = criterion(logits, batch_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = logits.max(1)
        train_total += batch_labels.size(0)
        train_correct += predicted.eq(batch_labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*train_correct/train_total:.2f}%'
        })

        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

    train_acc = 100. * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    # Validate every 2 epochs
    should_validate = (epoch == 0) or ((epoch + 1) % 2 == 0) or (epoch == epochs - 1)

    if should_validate:
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, dynamic_ncols=True)

            for batch_input, batch_labels in pbar_val:
                batch_input = batch_input.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)

                logits, _ = model(batch_input)
                loss = criterion(logits, batch_labels)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()

                pbar_val.set_postfix({'acc': f'{100.*val_correct/val_total:.2f}%'})

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%")
        print(f"           Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
            del checkpoint_data
    else:
        print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}% (validation skipped)")

    # Always save latest checkpoint
    latest_path = "/content/drive/MyDrive/VoxCeleb1_WebDataset/models/optimized_mamba_latest.pth"
    latest_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': best_val_acc,
        'train_acc': train_acc
    }
    torch.save(latest_checkpoint, latest_path)
    del latest_checkpoint

    torch.cuda.empty_cache()

# Save final model
final_path = "/content/drive/MyDrive/VoxCeleb1_WebDataset/models/optimized_mamba_final.pth"
torch.save(model.state_dict(), final_path)

print(f"\n{'='*70}")
print(f"✓ TRAINING COMPLETE!")
print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"{'='*70}")

torch.cuda.empty_cache()
