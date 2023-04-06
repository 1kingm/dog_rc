# properties of the data frame
VisionMaxLen = 16
VisionDataLen = 13

# the defination of the meaning of each byte
HEADER_BYTE = 0
X_BYTE = 2
Y_BYTE = 4
TASK_BYTE = 6
FLOAT_DATA_BYTE = 7
MOVE_ENABLE_BYTE = 11
# checksum byte should be the last byte of each frame
# this byte might be modified in the future since more bytes might be added
CHECKSUM_BYTE = 12
