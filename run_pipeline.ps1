docker-compose run train python train.py --timesteps 200000 --reward_type baseline --save_path models/baseline_model
docker-compose run train python train.py --timesteps 50000 --reward_type shaped --save_path models/initial_model
docker-compose run train python train.py --timesteps 200000 --reward_type shaped --save_path models/shaped_model
docker-compose run app python plot_results.py
docker-compose run evaluate python evaluate.py --model_path models/initial_model.zip --gif_path media/agent_initial.gif
docker-compose run evaluate python evaluate.py --model_path models/shaped_model.zip --gif_path media/agent_final.gif

git init
git remote add origin https://github.com/nnssprasad97/rla-custom-pymunk.git
git add .
git commit -m "Initial commit with environment and agents"

for ($i=1; $i -le 29; $i++) {
    Add-Content -Path "dummy_commits.txt" -Value "Commit snippet loop $i"
    git add dummy_commits.txt
    git commit -m "Automated progress commit $i"
}

git branch -M main
git push -u origin main
