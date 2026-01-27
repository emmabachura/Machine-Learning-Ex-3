1. SETUP (Virtual Environment)
------------------------------

# 1. Create the environment
python3 -m venv venv

# 2. Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# 3. Install dependencies from the frozen requirement file
pip install -r requirements.txt

2. HOW TO RUN
-------------

Command:
python src/Path-Finding.py --grid [1,2,3] --episodes [N]

Parameters:
--grid      : 1 (Simple), 2 (Zig-Zag), or 3 (Bottleneck)
--episodes  : Number of training trials (e.g., 5000)
--gamma     : Discount factor (Default: 0.95)
--max_steps : Maximum steps in an episode (Default 3000)
--seed      : Random seed for reproducibility (Default: 123)


Example:
python3 src/Path-Finding.py --grid 3 --episodes 1000 --seed 123