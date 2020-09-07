import numpy as np
import torch
from typing import Optional, Sequence


class MLP(torch.nn.Module):

    def __init__(self,
                 layer_sizes: Sequence[int],
                 input_size: int,
                 num_classes: int = 2,
                 learning_rate: float = 2.5e-4):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        previous_size = input_size
        for size in layer_sizes:
            self.layers.append(torch.nn.Linear(in_features=previous_size, out_features=size))
            self.layers.append(torch.nn.Tanh())
            previous_size = size
        self.layers.append(torch.nn.Linear(in_features=previous_size, out_features=num_classes))

        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if torch.cuda.is_available():
            self.cuda()

    def predict(self, input: np.ndarray, target: Optional[np.ndarray] = None) -> np.ndarray:
        input = torch.tensor(input, dtype=torch.float32)
        if target is not None:
            target = torch.tensor(target)
        if torch.cuda.is_available():
            input = input.cuda()
            if target is not None:
                target = target.cuda()

        # Prediction
        for layer in self.layers:
            input = layer(input)
        output = input.argmax(dim=1)

        # Update
        if target is not None:
            self.optimizer.zero_grad()
            loss = self.loss(input, target)
            loss.backward()
            self.optimizer.step()

        if torch.cuda.is_available():
            output = output.cpu()
        return output.numpy()
