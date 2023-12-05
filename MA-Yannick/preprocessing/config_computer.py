from enum import IntEnum
import socket


class ComputerID(IntEnum):
    YANNICK_LAPTOP = 0
    MATZE_IPE_PC = 1
    MATZE_IPE_LAPTOP = 2
    IPE_AVT_SIM = 3


hostname = socket.gethostname()
if hostname is None:
    computer = ComputerID.YANNICK_LAPTOP
elif hostname == 'IPELUHNB':
    computer = ComputerID.MATZE_IPE_LAPTOP
elif hostname == 'IPEBLANK-PC6':
    computer = ComputerID.MATZE_IPE_PC
elif hostname == 'IPEAVTSIM':
    computer = ComputerID.IPE_AVT_SIM
else:
    computer = ComputerID.YANNICK_LAPTOP
