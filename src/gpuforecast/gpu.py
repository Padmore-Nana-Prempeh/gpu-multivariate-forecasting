from __future__ import annotations

import torch


class CUDAPrefetcher:
    """Moves the next CPU batch to CUDA on a dedicated stream."""

    def __init__(self, loader, device: torch.device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_x = None
        self.next_y = None
        self._preload()

    def _preload(self) -> None:
        try:
            x, y = next(self.loader)
        except StopIteration:
            self.next_x = self.next_y = None
            return
        with torch.cuda.stream(self.stream):
            self.next_x = x.to(self.device, non_blocking=True)
            self.next_y = y.to(self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_x is None:
            raise StopIteration
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        x, y = self.next_x, self.next_y
        x.record_stream(torch.cuda.current_stream(self.device))
        y.record_stream(torch.cuda.current_stream(self.device))
        self._preload()
        return x, y
