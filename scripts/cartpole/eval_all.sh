set -e
mkdir -p logs/eval

python -m src.eval_cp --model_path models/ppo_cartpole_solver_seed7.zip   --algo ppo --episodes 100 --persona solver --csv_out logs/eval/ppo_cartpole_solver_seed7.csv
python -m src.eval_cp --model_path models/a2c_cartpole_solver_seed7.zip   --algo a2c --episodes 100 --persona solver --csv_out logs/eval/a2c_cartpole_solver_seed7.csv
python -m src.eval_cp --model_path models/ppo_cartpole_fuzzer_seed7.zip   --algo ppo --episodes 100 --persona fuzzer --csv_out logs/eval/ppo_cartpole_fuzzer_seed7.csv
python -m src.eval_cp --model_path models/a2c_cartpole_fuzzer_seed7.zip   --algo a2c --episodes 100 --persona fuzzer --csv_out logs/eval/a2c_cartpole_fuzzer_seed7.csv
