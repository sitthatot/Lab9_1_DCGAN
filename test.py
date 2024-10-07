import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

# Generator Network
class Generator(nn.Module):
    def __init__(self, z_dim=100, channels_img=3, features_g=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
            self._block(features_g * 2, features_g, 4, 2, 1),  # 64x64
            nn.ConvTranspose2d(
                features_g, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, channels_img=3, features_d=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            self._block(features_d * 8, features_d * 16, 4, 2, 1),
            nn.Conv2d(features_d * 16, 1, 4, 2, 0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

# Initialize weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# Training function
def train(dataloader, gen, disc, opt_gen, opt_disc, criterion, fixed_noise, num_epochs, device, save_dir):
    writer_real = SummaryWriter(f"{save_dir}/logs/real")
    writer_fake = SummaryWriter(f"{save_dir}/logs/fake")
    step = 0
    
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            batch_size = real.shape[0]
            noise = torch.randn(batch_size, Z_DIM, 1, 1).to(device)
            
            # Train Discriminator
            disc.zero_grad()
            fake = gen(noise)
            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            gen.zero_grad()
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            loss_gen.backward()
            opt_gen.step()

            # Print progress
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} "
                    f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    img_grid_real = make_grid(real[:32], normalize=True)
                    img_grid_fake = make_grid(fake[:32], normalize=True)
                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                    
                    # Save sample images
                    save_image(fake[:25], f"{save_dir}/samples/fake_{epoch}_{batch_idx}.png",
                             normalize=True, nrow=5)
                step += 1

        # Save model checkpoint
        torch.save({
            'generator_state_dict': gen.state_dict(),
            'discriminator_state_dict': disc.state_dict(),
            'gen_optimizer_state_dict': opt_gen.state_dict(),
            'disc_optimizer_state_dict': opt_disc.state_dict(),
        }, f"{save_dir}/checkpoints/checkpoint_epoch_{epoch}.pth")

# Function to generate and display images
def generate_and_display(generator, z_dim=100, num_images=25, device='cpu'):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, z_dim, 1, 1).to(device)
        generated_images = generator(noise)
        generated_images = (generated_images + 1) / 2  # Denormalize
        
    grid = make_grid(generated_images.cpu(), nrow=5, normalize=False)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

# Main training setup and execution
if __name__ == "__main__":
    # Hyperparameters
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 128
    IMAGE_SIZE = 128
    CHANNELS_IMG = 3
    Z_DIM = 100
    NUM_EPOCHS = 2
    FEATURES_D = 64
    FEATURES_G = 64
    
    # Directory setup
    DATA_DIR = "data"  # Change this to your dataset directory
    SAVE_DIR = "results"
    os.makedirs(f"{SAVE_DIR}/checkpoints", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/samples", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/logs", exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preprocessing
    transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Dataset and dataloader
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Initialize networks
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_G).to(device)
    disc = Discriminator(CHANNELS_IMG, FEATURES_D).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    # Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Fixed noise for visualization
    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

    # Train the model
    train(dataloader, gen, disc, opt_gen, opt_disc, criterion, fixed_noise, 
          NUM_EPOCHS, device, SAVE_DIR)

    # Generate sample images after training
    generate_and_display(gen, Z_DIM, num_images=25, device=device)