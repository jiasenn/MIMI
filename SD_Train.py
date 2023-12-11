import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import StableDiffusionPanoramaPipeline, DDIMScheduler
from PIL import Image
import os

# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.image_folder))

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, f"image_{idx}.jpg")
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image

# Set up data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust size as needed
    transforms.ToTensor(),
])

# Set up dataset and dataloader
dataset = CustomDataset(image_folder="/path/to/your/images", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# Load the pre-trained model and initialize the pipeline
model_ckpt = "stabilityai/stable-diffusion-2-base"
scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
pipe = StableDiffusionPanoramaPipeline.from_pretrained(
    model_ckpt, scheduler=scheduler, torch_dtype=torch.float32
)

# Set up the optimizer and criterion
optimizer = torch.optim.Adam(pipe.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        batch = batch.to(device)

        # Forward pass
        output = pipe(batch)

        # Compute loss
        loss = criterion(output, batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the fine-tuned model
pipe.save_pretrained("fine_tuned_model")
