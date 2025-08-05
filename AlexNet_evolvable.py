# AlexNet_evolvable.py
import torch
import torch.nn as nn
import random

# This dictionary defines the "genes" and the possible values (alleles) they can take.
# This is our Search Space. The GA will pick values from here to build a chromosome.
SEARCH_SPACE = {
    # Convolutional Layers
    'conv1_filters': [32, 64, 96],
    'conv1_kernel': [7, 9, 11],
    'conv1_stride': [3, 4],  # Removed stride 5 as it's too aggressive for 32x32 images
    'conv2_filters': [128, 192, 256],
    'conv2_kernel': [3, 5],
    'conv3_filters': [256, 384, 440],
    'conv4_filters': [256, 384],
    'conv5_filters': [192, 256],
    # Fully Connected Layers
    'fc1_neurons': [2048, 3072, 4096],
    'fc2_neurons': [2048, 3072, 4096],
    # Hyperparameters
    'lr': [0.001, 0.005, 0.01],
    'momentum': [0.85, 0.9, 0.95],
    'dropout': [0.4, 0.5, 0.6],
}

def create_random_chromosome():
    """
    Creates a random chromosome by picking a random value for each gene from the search space.
    This is used to create the initial population.
    """
    chromosome = {}
    for key, values in SEARCH_SPACE.items():
        chromosome[key] = random.choice(values)
    return chromosome

def supported_hyperparameters():
    """Kept for compatibility with the nn-dataset framework (if used)."""
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        # Use the learning rate and momentum from our chromosome
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()  # Set the model to training mode
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, chromosome: dict, device: torch.device):
        """
        The __init__ method now takes a 'chromosome' dictionary to build the network.
        """
        super().__init__()
        self.device = device
        self.chromosome = chromosome  # Store the chromosome for reference

        # ---- Feature Extractor (Convolutional Layers) ----
        # We build this layer by layer to handle the changing number of input/output channels.
        layers = []
        # CORRECTED: Use in_shape[0] for input channels (e.g., 3 for RGB images)
        in_channels = in_shape[0]

        # Layer 1
        layers += [
            nn.Conv2d(in_channels, chromosome['conv1_filters'], kernel_size=chromosome['conv1_kernel'],
                      stride=chromosome['conv1_stride'], padding=2),
            nn.ReLU(inplace=True),
            # CORRECTED: MaxPool2d kernel_size changed from 3 to 2 for smaller images
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = chromosome['conv1_filters']

        # Layer 2
        layers += [
            nn.Conv2d(in_channels, chromosome['conv2_filters'], kernel_size=chromosome['conv2_kernel'], padding=2),
            nn.ReLU(inplace=True),
            # CORRECTED: MaxPool2d kernel_size changed from 3 to 2 for smaller images
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = chromosome['conv2_filters']

        # Layer 3
        layers += [
            nn.Conv2d(in_channels, chromosome['conv3_filters'], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        in_channels = chromosome['conv3_filters']

        # Layer 4
        layers += [
            nn.Conv2d(in_channels, chromosome['conv4_filters'], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        in_channels = chromosome['conv4_filters']

        # Layer 5
        layers += [
            nn.Conv2d(in_channels, chromosome['conv5_filters'], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # CORRECTED: MaxPool2d kernel_size changed from 3 to 2 for smaller images
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = chromosome['conv5_filters']

        self.features = nn.Sequential(*layers)

        # This is a crucial trick! It ensures the output of the conv layers is always a fixed
        # size (6x6), regardless of the exact kernel sizes and strides we used.
        # This means we don't have to calculate the complex output shape of the conv block.
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # ---- Classifier (Fully Connected Layers) ----
        dropout_p = chromosome['dropout']
        # The input to the first Linear layer depends on the number of filters in the LAST conv layer.
        classifier_input_features = in_channels * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(classifier_input_features, chromosome['fc1_neurons']),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(chromosome['fc1_neurons'], chromosome['fc2_neurons']),
            nn.ReLU(inplace=True),
            nn.Linear(chromosome['fc2_neurons'], out_shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --- Add this helper function at the end of AlexNet_evolvable.py ---

def generate_model_code_string(chromosome: dict, in_shape: tuple = (3, 32, 32), out_shape: tuple = (10,)) -> str:
    """
    Generates the Python code string for the Net class based on the chromosome.
    This is used for duplicate checking and saving unique architectures.
    Assumes in_shape=(3, 32, 32) and out_shape=(10) for CIFAR-10 unless specified.
    """
    # Import statements for the generated file
    code_lines = [
        "import torch",
        "import torch.nn as nn",
        "",
        "class Net(nn.Module):",
        "    def __init__(self, in_shape, out_shape, device):",
        f"        super().__init__()",
        f"        self.device = device",
        # --- Feature Extractor ---
        "        layers = []",
        f"        in_channels = {in_shape[0]}", # Input channels (e.g., 3 for RGB)
        "",
        # Layer 1
        f"        layers += [",
        f"            nn.Conv2d(in_channels, {chromosome['conv1_filters']}, kernel_size={chromosome['conv1_kernel']},",
        f"                      stride={chromosome['conv1_stride']}, padding=2),",
        f"            nn.ReLU(inplace=True),",
        f"            nn.MaxPool2d(kernel_size=2, stride=2),",
        f"        ]",
        f"        in_channels = {chromosome['conv1_filters']}",
        "",
        # Layer 2
        f"        layers += [",
        f"            nn.Conv2d(in_channels, {chromosome['conv2_filters']}, kernel_size={chromosome['conv2_kernel']}, padding=2),",
        f"            nn.ReLU(inplace=True),",
        f"            nn.MaxPool2d(kernel_size=2, stride=2),",
        f"        ]",
        f"        in_channels = {chromosome['conv2_filters']}",
        "",
        # Layer 3
        f"        layers += [",
        f"            nn.Conv2d(in_channels, {chromosome['conv3_filters']}, kernel_size=3, padding=1),",
        f"            nn.ReLU(inplace=True),",
        f"        ]",
        f"        in_channels = {chromosome['conv3_filters']}",
        "",
        # Layer 4
        f"        layers += [",
        f"            nn.Conv2d(in_channels, {chromosome['conv4_filters']}, kernel_size=3, padding=1),",
        f"            nn.ReLU(inplace=True),",
        f"        ]",
        f"        in_channels = {chromosome['conv4_filters']}",
        "",
        # Layer 5
        f"        layers += [",
        f"            nn.Conv2d(in_channels, {chromosome['conv5_filters']}, kernel_size=3, padding=1),",
        f"            nn.ReLU(inplace=True),",
        f"            nn.MaxPool2d(kernel_size=2, stride=2),",
        f"        ]",
        f"        in_channels = {chromosome['conv5_filters']}",
        "        self.features = nn.Sequential(*layers)",
        "        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))",
        "",
        # --- Classifier ---
        f"        dropout_p = {chromosome['dropout']}",
        f"        classifier_input_features = in_channels * 6 * 6", # 6x6 from AdaptiveAvgPool2d
        "        self.classifier = nn.Sequential(",
        "            nn.Dropout(p=dropout_p),",
        f"            nn.Linear(classifier_input_features, {chromosome['fc1_neurons']}),",
        "            nn.ReLU(inplace=True),",
        "            nn.Dropout(p=dropout_p),",
        f"            nn.Linear({chromosome['fc1_neurons']}, {chromosome['fc2_neurons']}),",
        "            nn.ReLU(inplace=True),",
        f"            nn.Linear({chromosome['fc2_neurons']}, out_shape[0]),", # Output classes
        "        )",
        "",
        "    def forward(self, x):",
        "        x = self.features(x)",
        "        x = self.avgpool(x)",
        "        x = torch.flatten(x, 1)",
        "        x = self.classifier(x)",
        "        return x",
        "",
        "    # --- Methods required by ab.nn.api.check_nn or general utility ---",
        "    def train_setup(self, prm):",
        "        self.to(self.device)",
        "        self.criteria = (nn.CrossEntropyLoss().to(self.device),)",
        "        self.optimizer = torch.optim.SGD(",
        "            self.parameters(),",
        f"            lr=prm['lr'],", # Use lr from prm
        f"            momentum=prm['momentum']", # Use momentum from prm
        "        )",
        "",
        "    def learn(self, train_data):",
        "        self.train()",
        "        for inputs, labels in train_data:",
        "            inputs, labels = inputs.to(self.device), labels.to(self.device)",
        "            self.optimizer.zero_grad()",
        "            outputs = self(inputs)",
        "            loss = self.criteria[0](outputs, labels)",
        "            loss.backward()",
        "            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)",
        "            self.optimizer.step()",
        "",
        "# --- Chromosome used to generate this model ---",
        f"# Chromosome: {chromosome}",
        ""
    ]
    return "\n".join(code_lines)
