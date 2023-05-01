import taichi as ti
ti.init(arch=ti.cuda, device_memory_GB=6)
from .volume_render_test import composite_test