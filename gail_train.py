
from gail import Generator, GAIL, Discriminator

# gail_obj = GAIL('CartPole-v0')
gail_obj = GAIL("Acrobot-v1")

# gail_obj.get_demonstrations('CartPole-v0)
gail_obj.train()
