import numpy as np
np.random.seed(seed=42)
mach_tabe = np.arange(10)
for i in range(10):
    np.random.shuffle(mach_tabe)
    print(mach_tabe)


np.random.seed(seed=42)
mach_tabe = np.arange(10)
np.resize(mach_tabe,[-1,2])

for i in range(10):
    np.random.shuffle(mach_tabe)
    print(mach_tabe)