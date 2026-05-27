"""
Wrapper around the gymnasium Acrobot environment to modify its dynamics for transfer learning experiments.

This wrapper allows testing RL policy robustness by perturbing the physical parameters of the Acrobot.
You can use a single `asymmetry` parameter to gradually change the dynamics, or specify
individual link lengths and masses for fine-grained control.

Key Changes from Original Acrobot:
1. Updates LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2
2. Recalculates center of mass positions (LINK_COM_POS_1, LINK_COM_POS_2)
3. Computes separate moment of inertia for each link (LINK_MOI_1, LINK_MOI_2)
4. Overrides _dsdt() method to use separate MOI values (original uses single LINK_MOI for both)

The `asymmetry` parameter provides a single dial for transfer experiments:
- asymmetry=1.0: Original symmetric Acrobot (both links identical)
- asymmetry>1.0: Second link becomes heavier and longer (harder to swing up)
- asymmetry<1.0: Second link becomes lighter and shorter (easier to swing up)

This is analogous to wind speed in LunarLander transfer experiments.
"""
import gymnasium as gym
from gymnasium.envs.classic_control import AcrobotEnv

class AcrobotTransferEnv(gym.Wrapper):

    def __init__(
        self,
        env: AcrobotEnv,
        asymmetry: float | None = None,
        link_lengths: tuple[float, float] | None = None,
        link_masses: tuple[float, float] | None = None,
    ):
        """
        Initialize the AcrobotTransferEnv wrapper.

        Args:
            env: The original Acrobot environment to wrap.
            asymmetry: Single parameter controlling the ratio of link2/link1 properties.
                       At 1.0, both links are identical (original Acrobot).
                       Values > 1.0 make link2 heavier/longer, < 1.0 make it lighter/shorter.
                       This is the recommended parameter for transfer experiments.
            link_lengths: A tuple specifying the lengths of the two links.
                          Ignored if asymmetry is provided.
            link_masses: A tuple specifying the masses of the two links.
                         Ignored if asymmetry is provided.
        """
        super().__init__(env)

        # Store asymmetry for reference
        self.asymmetry = asymmetry

        # Get the actual AcrobotEnv (not intermediate wrappers)
        acrobot_env = self.unwrapped

        # Determine link parameters based on input mode
        lengths: tuple[float, float]
        masses: tuple[float, float]
        if asymmetry is not None:

            print("Use asymmetry parameter to set link lengths and masses:", asymmetry)

            # Single-parameter mode: derive lengths and masses from asymmetry
            # Link 1 stays at default (1.0), Link 2 scales by asymmetry
            lengths = (1.0, asymmetry)
            masses = (1.0, asymmetry)
        else:
            # Explicit mode: use provided values or defaults
            lengths = link_lengths if link_lengths is not None else (1.0, 1.0)
            masses = link_masses if link_masses is not None else (1.0, 1.0)

        # Modified parameters on the actual AcrobotEnv
        acrobot_env.LINK_LENGTH_1 = lengths[0]
        acrobot_env.LINK_LENGTH_2 = lengths[1]
        acrobot_env.LINK_MASS_1 = masses[0]
        acrobot_env.LINK_MASS_2 = masses[1]

        # Update center of mass positions (original defaults to 0.5 for both)
        # Assuming uniform density, COM is at the center of each link
        acrobot_env.LINK_COM_POS_1 = lengths[0] / 2.0
        acrobot_env.LINK_COM_POS_2 = lengths[1] / 2.0

        # Calculate separate moment of inertia for each link
        # Original uses single LINK_MOI=1.0 for both links in _dsdt
        # For uniform rods: I = (1/12) * m * L^2
        acrobot_env.LINK_MOI_1 = (1.0 / 12.0) * masses[0] * lengths[0]**2
        acrobot_env.LINK_MOI_2 = (1.0 / 12.0) * masses[1] * lengths[1]**2

        acrobot_env.LINK_MOI = acrobot_env.LINK_MOI_1

        # CHANGE 4: Override _dsdt to use separate MOI values (see method below)
        self._override_dsdt()

    def _override_dsdt(self):
        """
        directly adapdted from: https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/classic_control/acrobot.py#L237C21-L237C32

        Override the _dsdt method to use separate MOI values for each link.

        CHANGE FROM ORIGINAL: The original Acrobot uses self.LINK_MOI for both I1 and I2.
        This override allows each link to have its own moment of inertia (LINK_MOI_1, LINK_MOI_2),
        enabling asymmetric perturbations for robustness testing.

        The rest of the dynamics equations remain identical to the original implementation.
        """
        from numpy import cos, sin, pi

        # Capture reference to the actual AcrobotEnv
        acrobot_env = self.unwrapped

        def new_dsdt(s_augmented):
            m1 = acrobot_env.LINK_MASS_1
            m2 = acrobot_env.LINK_MASS_2
            l1 = acrobot_env.LINK_LENGTH_1
            lc1 = acrobot_env.LINK_COM_POS_1
            lc2 = acrobot_env.LINK_COM_POS_2
            I1 = acrobot_env.LINK_MOI_1  # Use separate MOI for link 1
            I2 = acrobot_env.LINK_MOI_2  # Use separate MOI for link 2
            g = 9.8
            a = s_augmented[-1]
            s = s_augmented[:-1]
            theta1 = s[0]
            theta2 = s[1]
            dtheta1 = s[2]
            dtheta2 = s[3]
            d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
            d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
            phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
            phi1 = (
                -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
                - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
                + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
                + phi2
            )
            if acrobot_env.book_or_nips == "nips":
                ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
            else:
                ddtheta2 = (
                    a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2
                ) / (m2 * lc2**2 + I2 - d2**2 / d1)
            ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
            return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0

        acrobot_env._dsdt = new_dsdt