# IEEF: An Interactive Evaluation Framework for Empathetic Response Generation

## Supervised Learning

Seq-to-seq training based on t5 model

### User Simulator Supvervised Training

```
python main.py -agent_type us -run_type train -ururu -backbone t5-large -model_dir simulator_t5_small -epoch 20
```

### Dialogue System Supervised Training
```
python main.py -agent_type ds -run_type train -ururu -backbone t5-large -model_dir dialogue_t5_small -epoch 20
```

### Interaction
Conduct interactions between a user simulator and a dialogue system (either SL-based models or RL-based models). Generate dialogue sessions based on user goals from test or dev set. 
```
python interact.py -simulator_path ./your_simulator_model_dir/checkpoint -dialog_sys_path ./your_dialogue_model_dir/your_checkpoint -model_name mttod -generate_results_path output.json
```

## Reinforcement Learning
```
python interact.py -do_rl_training -seed 1998 -simulator_save_path simulator_rl -dialog_save_path dialog_rl
```

## Interactive Evaluation
First generate dialogue by interactions between a user simulator and a dialogue system. Then scoring dialogue logs by LLM.

```
python compute_all_scores.py -output_result_path output.json -config_dir dialogue_t5_small
```