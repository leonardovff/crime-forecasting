# tcc-learning

## Troubleshooting

### GPU not available — `torch.cuda.is_available()` returns `False`

Symptoms:

```
UserWarning: CUDA initialization: CUDA unknown error ...
  return torch._C._cuda_getDeviceCount() > 0
Should return True if a GPU is available:  False
Number of GPUs available:  1
```

`device_count()` reads NVML (no runtime init) so it still reports `1`, while
`is_available()` fails during actual CUDA init with error `999`
(`CUDA_ERROR_UNKNOWN`). This is a driver-state issue, not a code bug —
typically after suspend/resume or a driver update leaves `nvidia_uvm` in a
broken state.

Fix: reload the UVM kernel module, then restart the notebook kernel.

```sh
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
```

If it still fails, reboot.