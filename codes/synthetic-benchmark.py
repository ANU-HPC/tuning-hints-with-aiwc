#!env python3

import glob
from subprocess import Popen, PIPE

usable_kernels = []
unusable_kernels = []

print("Checking next kernel batch...")
synthetic_kernel_batch = glob.glob("./sample_kernels/*.cl")
for synthetic_kernel in synthetic_kernel_batch:
    process = Popen(["./sbc", synthetic_kernel, "tiny", "0", "0"], stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if exit_code == 0: #a usable kernel added it to the list to test for prediction
        usable_kernels.append(synthetic_kernel)
    else:
        unusable_kernels.append(synthetic_kernel)
print("Passed {} and failed {} kernels.".format(len(usable_kernels),len(unusable_kernels)))

print("Generating payload AIWC metrics...")
for kernel in usable_kernels:
    #TODO: add for loop for every problem size
    process = Popen(["oclgrind", "--workload-characterisation", "./sbd", kernel, "tiny", "0", "0", "aiwc"], stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if exit_code == 0:
        print(output)
    
print("Querying model...")

print("Generating payload runtimes...")

