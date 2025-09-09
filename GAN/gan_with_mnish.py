import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# -----------------------------
# Generator Network
# -----------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_dim),
            nn.Tanh()  # scale outputs between -1 and 1
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------
# Discriminator Network
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # probability of being real
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------
# GAN Trainer Class
# -----------------------------
class GAN:
    def __init__(self, z_dim=64, img_dim=28*28, lr=0.0002, batch_size=128, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.z_dim = z_dim
        self.img_dim = img_dim
        self.batch_size = batch_size

        # Initialize models
        self.gen = Generator(z_dim, img_dim).to(self.device)
        self.disc = Discriminator(img_dim).to(self.device)

        # Optimizers
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=lr)
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=lr)

        # Loss function
        self.criterion = nn.BCELoss()

        # Fixed noise for visualization
        self.fixed_noise = torch.randn(16, z_dim).to(self.device)

        # Data loader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # scale to [-1, 1]
        ])
        dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, epochs=50):
        for epoch in range(epochs):
            for real, _ in self.loader:
                real = real.view(-1, self.img_dim).to(self.device)
                batch_size = real.size(0)

                # Labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # -----------------------------
                # Train Discriminator
                # -----------------------------
                noise = torch.randn(batch_size, self.z_dim).to(self.device)
                fake = self.gen(noise)

                disc_real = self.disc(real)
                disc_fake = self.disc(fake.detach())

                loss_disc_real = self.criterion(disc_real, real_labels)
                loss_disc_fake = self.criterion(disc_fake, fake_labels)
                loss_disc = loss_disc_real + loss_disc_fake

                self.opt_disc.zero_grad()
                loss_disc.backward()
                self.opt_disc.step()

                # -----------------------------
                # Train Generator
                # -----------------------------
                noise = torch.randn(batch_size, self.z_dim).to(self.device)
                fake = self.gen(noise)
                output = self.disc(fake)

                loss_gen = self.criterion(output, real_labels)  # want fake → real

                self.opt_gen.zero_grad()
                loss_gen.backward()
                self.opt_gen.step()

            # -----------------------------
            # Logging
            # -----------------------------
            print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {loss_disc:.4f} | Loss G: {loss_gen:.4f}")
            if (epoch+1) % 10 == 0:
                self.visualize(epoch+1)

    def visualize(self, epoch):
        """Show generated images for monitoring progress"""
        with torch.no_grad():
            fake_images = self.gen(self.fixed_noise).reshape(-1, 1, 28, 28)
            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(fake_images[i].cpu().squeeze(), cmap="gray")
                ax.axis("off")
            plt.suptitle(f"Epoch {epoch}")
            plt.show()


# -----------------------------
# Run the GAN
# -----------------------------
if __name__ == "__main__":
    gan = GAN()
    gan.train(epochs=50)
