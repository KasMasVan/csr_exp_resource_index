mkdir data/big_bench/
cd data/big_bench/

proxy_prefix="https://ghproxy.com/"
repo_prefix="https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/"
single_tasks=("ruin_names" "temporal_sequences" "emoji_movie")
conceptual_combinations_subtasks=("contradictions" "emergent_properties" "fanciful_fictional_combinations" "homonyms" "invented_words" "surprising_uncommon_combinations")
strange_stories_subtasks=("boolean" "multiple_choice")

for task in "${single_tasks[@]}"
do
    wget -O "${task}.json" "${proxy_prefix}${repo_prefix}${task}/task.json"
done

for subtask in "${conceptual_combinations_subtasks[@]}"
do
    wget -O "conceptual_combinations_${subtask}.json" "${proxy_prefix}${repo_prefix}conceptual_combinations/${subtask}/task.json"
done

for subtask in "${strange_stories_subtasks[@]}"
do
    wget -O "strange_stories_${subtask}.json" "${proxy_prefix}${repo_prefix}strange_stories/${subtask}/task.json"
    
done
