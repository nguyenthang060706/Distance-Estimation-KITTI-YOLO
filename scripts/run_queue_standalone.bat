@echo off
title Detector Training Queue (D7: epochs=100, patience=100)
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "d:\Tài liệu học\FALL2026\DSR301m\Distance-Estimation-KITTI-YOLO"
echo ======================================================================
echo Launching 3-detector training queue from scratch (D7 protocol)
echo GPU: NVIDIA RTX 5060 (CUDA 12.8)
echo Python: C:\Users\ADMIN\AppData\Local\Python\bin\python.exe
echo ======================================================================
C:\Users\ADMIN\AppData\Local\Python\bin\python.exe scripts\train_queue.py
echo ======================================================================
echo Queue completed or terminated.
echo ======================================================================
pause
