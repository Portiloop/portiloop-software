from abc import ABC, abstractmethod
import numpy as np
from scipy import signal

def filter_24(value):
    return (value * 4.5) / (2**23 - 1) / 24.0 * 1e6  # 23 because 1 bit is lost for sign


def filter_2scomplement_np(value):
    return np.where((value & (1 << 23)) != 0, value - (1 << 24), value)


def int_to_float(value):
    """
    Convert the int value out of the ADS into a value in microvolts
    """
    return filter_24(filter_2scomplement_np(value))


def shift_numpy(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


class FIR:
    def __init__(self, nb_channels, coefficients, buffer=None):

        self.coefficients = np.expand_dims(np.array(coefficients), axis=1)
        self.taps = len(self.coefficients)
        self.nb_channels = nb_channels
        self.buffer = (
            np.array(buffer)
            if buffer is not None
            else np.zeros((self.taps, self.nb_channels))
        )

    def filter(self, x):
        self.buffer = shift_numpy(self.buffer, 1, x)
        filtered = np.sum(self.buffer * self.coefficients, axis=0)
        return filtered


class BaseFilter(ABC):
    _registry = {}
        
    def __init_subclass__(cls,  register_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        name = register_name or cls.__name__
        cls._registry[name] = cls

    @classmethod
    def get_filter(cls, name:str):
        return cls._registry.get(name)
    
    @abstractmethod
    def filter(self, value):
        raise NotImplementedError

class SpindleFilter(BaseFilter, register_name="Spindle"):
    def __init__(
        self,
        nb_channels,
        sampling_rate,
        power_line_fq=60,
        use_custom_fir=False,
        custom_fir_order=20,
        custom_fir_cutoff=30,
        alpha_avg=0.1,
        alpha_std=0.001,
        epsilon=0.000001,
        filter_args=[],
        verbose=False,
    ):
        if len(filter_args) > 0:
            use_fir, use_notch, use_std = filter_args
        else:
            use_fir = (True,)
            use_notch = (True,)
            use_std = True
        self.use_fir = use_fir
        self.use_notch = use_notch
        self.use_std = use_std
        self.nb_channels = nb_channels
        assert power_line_fq in [
            50,
            60,
        ], f"The only supported power line frequencies are 50 Hz and 60 Hz"
        if power_line_fq == 60:
            self.notch_coeff1 = -0.12478308884588535
            self.notch_coeff2 = 0.98729186796473023
            self.notch_coeff3 = 0.99364593398236511
            self.notch_coeff4 = -0.12478308884588535
            self.notch_coeff5 = 0.99364593398236511
        else:
            self.notch_coeff1 = -0.61410695998423581
            self.notch_coeff2 = 0.98729186796473023
            self.notch_coeff3 = 0.99364593398236511
            self.notch_coeff4 = -0.61410695998423581
            self.notch_coeff5 = 0.99364593398236511
        self.dfs = [np.zeros(self.nb_channels), np.zeros(self.nb_channels)]

        self.moving_average = None
        self.moving_variance = np.zeros(self.nb_channels)
        self.ALPHA_AVG = alpha_avg
        self.ALPHA_STD = alpha_std
        self.EPSILON = epsilon

        if use_custom_fir:
            self.fir_coef = signal.firwin(
                numtaps=custom_fir_order + 1, cutoff=custom_fir_cutoff, fs=sampling_rate
            )
        else:
            self.fir_coef = [
                0.001623780150148094927192721215192250384,
                0.014988684599373741992978104065059596905,
                0.021287595318265635502275046064823982306,
                0.007349500393709578957568417933998716762,
                -0.025127515717112181709014251396183681209,
                -0.052210507359822452833064687638398027048,
                -0.039273839505489904766477593511808663607,
                0.033021568427940004020193498490698402748,
                0.147606943281569008563636202779889572412,
                0.254000252034505602516389899392379447818,
                0.297330876398883392486283128164359368384,
                0.254000252034505602516389899392379447818,
                0.147606943281569008563636202779889572412,
                0.033021568427940004020193498490698402748,
                -0.039273839505489904766477593511808663607,
                -0.052210507359822452833064687638398027048,
                -0.025127515717112181709014251396183681209,
                0.007349500393709578957568417933998716762,
                0.021287595318265635502275046064823982306,
                0.014988684599373741992978104065059596905,
                0.001623780150148094927192721215192250384,
            ]
        self.fir = FIR(self.nb_channels, self.fir_coef)

        self.verbose = verbose

    def filter(self, value):
        """
        value: a numpy array of shape (data series, channels)
        """
        if self.verbose:
            print(f"Spindle Filtering shape {value.shape}")

        for i, x in enumerate(value):  # loop over the data series
            # FIR:
            if self.use_fir:
                x = self.fir.filter(x)
            # notch:
            if self.use_notch:
                denAccum = (
                    x - self.notch_coeff1 * self.dfs[0]
                ) - self.notch_coeff2 * self.dfs[1]
                x = (
                    self.notch_coeff3 * denAccum + self.notch_coeff4 * self.dfs[0]
                ) + self.notch_coeff5 * self.dfs[1]
                self.dfs[1] = self.dfs[0]
                self.dfs[0] = denAccum
            # standardization:
            if self.use_std:
                if self.moving_average is not None:
                    delta = x - self.moving_average
                    self.moving_average = self.moving_average + self.ALPHA_AVG * delta
                    self.moving_variance = (1 - self.ALPHA_STD) * (
                        self.moving_variance + self.ALPHA_STD * delta**2
                    )
                    moving_std = np.sqrt(self.moving_variance)
                    x = (x - self.moving_average) / (moving_std + self.EPSILON)
                else:
                    self.moving_average = x
            value[i] = x
        return value


class SlowWaveFilter(BaseFilter, register_name="SlowWave"):
    def __init__(self, nb_channels, sampling_rate, verbose=False, **kwargs):
        print('SOOnlineFiltering initialized')
        self.fs = sampling_rate
        self.nb_channels = nb_channels
        self.verbose = verbose

        # DC offset removal filter (high-pass filter)
        self.dc_b, self.dc_a = signal.butter(1, 0.5/(self.fs/2), 'high')
        
        # 60 Hz notch filter
        f0 = 60.0  # Notch frequency
        Q = 100.0  # Quality factor
        self.notch_b, self.notch_a = signal.iirnotch(f0, Q, self.fs)
        
        # FIR Bandpass filter (0.5 - 30 Hz)
        low = 0.5
        high = 30.0
        # TODO Change to use FIR class
        self.bp_b = signal.firwin(20, [low, high], pass_zero=False, window='hamming', fs = self.fs)
        # Initialize filter states for each channel
        self.dc_states = [signal.lfilter_zi(self.dc_b, self.dc_a) for _ in range(self.nb_channels)]
        self.notch_states = [signal.lfilter_zi(self.notch_b, self.notch_a) for _ in range(self.nb_channels)]
        self.bp_states = [np.zeros(len(self.bp_b) - 1) for _ in range(self.nb_channels)]

    def filter(self, value):
        """
        value: a numpy array of shape (data series, channels)
        """
        if self.verbose:
            print(f"SO Filtering shape {value.shape}")
        
        filtered_value = np.zeros_like(value)

        for i in range(self.nb_channels):
            channel_data = value[:, i]

            # Apply notch filter
            notched, self.notch_states[i] = signal.lfilter(self.notch_b, self.notch_a, channel_data, zi=self.notch_states[i])
            
            # Apply FIR bandpass filter
            bandpassed, self.bp_states[i] = signal.lfilter(self.bp_b, 1, notched, zi=self.bp_states[i])
            
            # Remove DC offset
            filtered, self.dc_states[i] = signal.lfilter(self.dc_b, self.dc_a, bandpassed, zi=self.dc_states[i])
            
            filtered_value[:, i] = filtered

        return filtered_value