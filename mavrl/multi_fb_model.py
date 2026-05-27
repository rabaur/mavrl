from torch import nn
from typing import Any
from mavrl.feedback.base import FeedbackModule
from mavrl.priors import kl_div_std_normal
from mavrl.encoder.reward_encoder import BaseRewardEncoder
from mavrl.losses import td_error_regularizer
from mavrl.types import FeedbackType
from mavrl.types import DataKey


class MultiFeedbackTypeModel(nn.Module):

    def __init__(
        self,
        encoder: BaseRewardEncoder,
        q_model: nn.Module,
        decoders: dict[FeedbackType, FeedbackModule],
    ):
        super().__init__()
        self.encoder = encoder
        self.q_model = q_model
        self.decoders = nn.ModuleDict(decoders)
    
    def forward(self, **kwargs) -> Any:

        # Unpack variables
        obs = kwargs[DataKey.OBS]
        next_obs = kwargs[DataKey.NEXT_OBS]
        action_feats = kwargs[DataKey.ACT_FEATS]
        valid = kwargs[DataKey.VALID]
        terminal = kwargs[DataKey.TERMINAL]
        acts_curr = kwargs[DataKey.ACTS].long()
        acts_next = kwargs[DataKey.NEXT_ACTS].long()
        gamma = kwargs[DataKey.GAMMA][0]
        fb_type = kwargs[DataKey.FEEDBACK_TYPE][0]

        r_mu, r_log_var = self.encoder(obs, action_feats, next_obs)
        r_mu = r_mu.squeeze(-1)
        r_log_var = r_log_var.squeeze(-1)
        r_samples = self.encoder.sample(r_mu, r_log_var)
        # Always mask terminal states from KL to prevent the prior from pushing
        # terminal rewards toward zero (which distorts the learned reward structure)
        kl_div = kl_div_std_normal(r_mu, r_log_var, valid, terminal=terminal)

        # Route to appropriate head
        head = self.decoders[fb_type]  # same feedback type for all samples per batch
        
        q_curr = self.q_model(obs)
        q_next = self.q_model(next_obs)
        td_error = td_error_regularizer(
            acts_curr=acts_curr,
            acts_next=acts_next,
            q_curr=q_curr,
            q_next=q_next,
            r_mu=r_mu,
            r_log_var=r_log_var,
            gamma=gamma,
            valid=valid,
            terminal=terminal,
        )

        metrics = {
            "q_value_max": q_curr.max(),
            "q_value_min": q_curr.min(),
        }

        nll = 0
        if fb_type == FeedbackType.PREF:
            prefs = kwargs[DataKey.PREFERENCE]
            beta = kwargs[DataKey.RATIONALITY][0]
            nll = head(r_samples, valid, beta, prefs)
        elif fb_type == FeedbackType.DEMO:
            beta = kwargs[DataKey.RATIONALITY][0]
            nll = head(acts_curr, q_curr, beta, valid)
        elif fb_type == FeedbackType.RATE:
            ratings = kwargs[DataKey.RATING]
            nll = head(r_samples, valid, ratings)
        elif fb_type == FeedbackType.STOP:
            stop_times = kwargs[DataKey.STOP_TIME]
            lambd = kwargs[DataKey.LAMBDA][0]
            regret_discount = kwargs[DataKey.REGRET_DISCOUNT][0]
            nll = head(q_curr, acts_curr, stop_times, lambd, regret_discount, valid)
        else:
            raise ValueError(f"Invalid feedback type: {fb_type.value}")

        # Create final output
        output = {
            "negative_log_likelihood": nll,
            "kl_divergence": kl_div,
            "td_error": td_error,
        }
        output.update(metrics)
        return output
