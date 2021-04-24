from . import tools
from ..random_explorer import RandomExplorer, RandomExplorerParams


class Agent(RandomExplorer, tools.Agent):
    pass


def test_selected_action_is_sampled_more_frequently():
    rep = RandomExplorerParams(init_ep=0.6, final_ep=0.01, decay_ep=1e-4)
    test_agent = Agent(ep=rep)
    actions = list(range(5))
    chosen_actions = {}
    for _ in range(1000):
        a = test_agent.ex_choose(actions, lambda x: x[0])
        if a not in chosen_actions:
            chosen_actions[a] = 1
        else:
            chosen_actions[a] += 1

    for a in chosen_actions:
        assert chosen_actions[a] > 0
        if a == 0:
            continue
        assert chosen_actions[0] > chosen_actions[a]

