from mavrl.types import FeedbackType
from mavrl.feedback.pref import PreferenceModule
from mavrl.feedback.demo import DemonstrationsDecoder
from mavrl.feedback.rate import RatingModule
from mavrl.feedback.stop import StopModule

def make_nll(fb_type: FeedbackType, **kwargs):
    if fb_type == FeedbackType.PREF:
        return PreferenceModule()
    elif fb_type == FeedbackType.DEMO:
        return DemonstrationsDecoder()
    elif fb_type == FeedbackType.RATE:
        return RatingModule()
    elif fb_type == FeedbackType.STOP:
        return StopModule()
    else:
        raise ValueError(f"Invalid feedback type: {fb_type.value}")