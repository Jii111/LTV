# ICLTV
## * run_baselin_all
  ```
(1) 주요 config 설정(모델, 태스크, num_shot 등)
I2CL/configs/config_baseline_all.py
(2) 실험 실행
cd I2CL
CUDA_VISIBLE_DEVICES=1 python I2CL/test/run_baseline_all.sh

(3) 결과 확인
I2CL/config['exp_name'] 경로에 결과 파일 저장되어 있음
  ```
