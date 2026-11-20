<b><u>AI Transform</u></b>

Neural Style Transfer using Adaptive Instance Normalization (AdaIN)

Transform ordinary images into artistic masterpieces using Deep Learning.

<br>
</div>

⸻

✨ About The Project

AI Transform is a deep learning-powered web application that brings artistic creativity and artificial intelligence together.

Built using Neural Style Transfer (NST) and Adaptive Instance Normalization (AdaIN), this project allows users to transform ordinary photographs into visually stunning artistic images.

The application takes two inputs:

* Content Image: The image whose structure and objects you want to preserve.
* Style Image: The image whose artistic patterns, textures, and colors you want to apply.

Using a pretrained VGG-based encoder and a decoder network, the application generates a stylized image that combines the content of one image with the artistic characteristics of another.

🚀 Key Features

Feature	Description
 Image Upload	Upload your own content and style images.
 Artistic Transformation	Apply artistic styles using deep learning.
 AdaIN Architecture	Uses Adaptive Instance Normalization for feature alignment.
 Fast Processing	Generates stylized images without traditional per-image iterative optimization.
 Web Interface	Simple and accessible browser-based application.
 Output Generation	View and save transformed images.

🧠 How Does It Work?

The project implements the Adaptive Instance Normalization (AdaIN) technique introduced by Huang and Belongie (2017).

Architecture Workflow

       CONTENT IMAGE             STYLE IMAGE
             │                       │
             ▼                       ▼
       ┌───────────┐           ┌───────────┐
       │   VGG     │           │   VGG     │
       │  Encoder  │           │  Encoder  │
       └───────────┘           └───────────┘
             │                       │
             ▼                       ▼
       Content Features         Style Features
             │                       │
             └──────────┬────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │      AdaIN       │
              │ Feature Alignment│
              └──────────────────┘
                        │
                        ▼
               Transformed Features
                        │
                        ▼
                ┌─────────────┐
                │   Decoder   │
                └─────────────┘
                        │
                        ▼
                 🎨 STYLIZED IMAGE

🔬 Understanding AdaIN

Adaptive Instance Normalization transfers the statistical properties of style features to content features.

Its mathematical formulation is:

AdaIN(x, y) = σ(y) × ((x − μ(x)) / σ(x)) + μ(y)

Where:

* x = Content feature representation.
* y = Style feature representation.
* μ(x) = Channel-wise mean of content features.
* σ(x) = Channel-wise standard deviation of content features.
* μ(y) = Channel-wise mean of style features.
* σ(y) = Channel-wise standard deviation of style features.

The core idea: AdaIN aligns the mean and variance of content features with those of style features, enabling arbitrary artistic style transfer.

⸻

🖼️ Results & Visualizations

Experience the transformation of ordinary images into artistic creations.

Content Image vs Style Image vs Generated Output

Content Image	Style Image	Generated Image
Add image here	Add image here	Add result here

Replace these placeholders with actual screenshots of your project.

⸻

🛠️ Tech Stack

Core Technologies

*  Python
*  PyTorch
*  VGG19-based Feature Extraction
*  Adaptive Instance Normalization (AdaIN)
*  Flask

Frontend & Image Processing

* HTML5
* CSS3
* JavaScript
* Pillow
* NumPy

⸻

📂 Project Structure

AI_TRANSFORM/
│
├── content_data/          # Content image resources
├── examples/              # Example images
├── style_data/            # Style image resources
│
├── templates/             # HTML templates
│
├── static/                # Static assets and uploads
│
├── utils/
│   ├── models.py          # Model architecture
│   └── utils.py           # Utility functions
│
├── app.py                 # Flask application
├── train.py               # Model training script
│
├── requirements.txt       # Project dependencies
├── Procfile               # Deployment configuration
└── README.md

⸻

⚙️ Installation & Setup

Follow these steps to run AI Transform locally.

1️⃣ Clone the Repository

git clone YOUR_REPOSITORY_URL
cd AI_TRANSFORM

2️⃣ Create a Virtual Environment

python -m venv venv

Activate it:

macOS / Linux

source venv/bin/activate

Windows

venv\Scripts\activate

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Configure Pretrained Model Weights

Download or obtain the required pretrained model weights and place them in the location expected by the application.

For example:

vgg_normalised.pth

Make sure all required model checkpoints are available before running the application.

5️⃣ Run the Application

python app.py

Open your browser:

http://127.0.0.1:5000/

You’re ready to transform your images!

⸻

⚖️ Traditional NST vs AdaIN

Parameter	Traditional NST	AdaIN
Approach	Iterative optimization	Feature statistics alignment
Processing	Computationally expensive	More efficient
Arbitrary Styles	Requires optimization per style	Supports arbitrary style images
Feature Transformation	Optimization-based	Mean and variance alignment
Inference	Generally slower	Generally faster

⸻

🔮 Future Improvements

The project can be extended with several exciting features:

* Video-based Neural Style Transfer.
* Real-time artistic image transformation.
* More artistic style collections.
* Responsive mobile-friendly interface.
* Cloud-based model hosting.
* GPU-accelerated inference.
* Integration of advanced style-transfer architectures.

⸻

Research & References

This project is inspired by the following research:

1. Adaptive Instance Normalization

Huang, X., & Belongie, S. (2017).

Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization.

IEEE International Conference on Computer Vision (ICCV).

2. VGG Architecture

Simonyan, K., & Zisserman, A. (2014).

Very Deep Convolutional Networks for Large-Scale Image Recognition.

⸻

👨‍💻 Developer

AI Transform — A Deep Learning project exploring the intersection of Artificial Intelligence, Computer Vision, and Digital Art.

⸻

<div align="center">

✨ Turning Images into Art with Artificial Intelligence ✨

</div>
