import numpy as np
from scipy.signal import firwin
from scipy import signal

from portiloop.src.core.processing import Processor


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
        self.buffer = np.array(buffer) if buffer is not None else np.zeros((self.taps, self.nb_channels))

    def filter(self, x):
        self.buffer = shift_numpy(self.buffer, 1, x)
        filtered = np.sum(self.buffer * self.coefficients, axis=0)
        return filtered


class Notch:
    def __init__(self, power_line_fq, nb_channels):
        if power_line_fq == 60:
            self.notch_coeff1 = -0.12478308884588535
            self.notch_coeff2 = 0.98729186796473023
            self.notch_coeff3 = 0.99364593398236511
            self.notch_coeff4 = -0.12478308884588535
            self.notch_coeff5 = 0.99364593398236511
        else:
            self.notch_coeff1 = -0.61410695998423581
            self.notch_coeff2 =  0.98729186796473023
            self.notch_coeff3 = 0.99364593398236511
            self.notch_coeff4 = -0.61410695998423581
            self.notch_coeff5 = 0.99364593398236511
        self.dfs = [np.zeros(nb_channels), np.zeros(nb_channels)]

    def filter(self, x):
        denAccum = (x - self.notch_coeff1 * self.dfs[0]) - self.notch_coeff2 * self.dfs[1]
        x = (self.notch_coeff3 * denAccum + self.notch_coeff4 * self.dfs[0]) + self.notch_coeff5 * self.dfs[1]
        self.dfs[1] = self.dfs[0]
        self.dfs[0] = denAccum
        return x


class Standardization:
    def __init__(self, nb_channels, alpha_avg, alpha_std, epsilon):
        self.moving_average = None
        self.moving_variance = np.zeros(nb_channels)
        self.ALPHA_AVG = alpha_avg
        self.ALPHA_STD = alpha_std
        self.EPSILON = epsilon

    def filter(self, x):
        if self.moving_average is not None:
            delta = x - self.moving_average
            self.moving_average = self.moving_average + self.ALPHA_AVG * delta
            self.moving_variance = (1 - self.ALPHA_STD) * (self.moving_variance + self.ALPHA_STD * delta ** 2)
            moving_std = np.sqrt(self.moving_variance)
            x = (x - self.moving_average) / (moving_std + self.EPSILON)
        else:
            self.moving_average = x
        return x


class DC:
    def __init__(self, dc_estimate, alpha):
        self.dc_estimate = dc_estimate
        self.alpha = alpha

    def filter(self, x):
        self.dc_estimate = (1 - self.alpha) * self.dc_estimate + self.alpha * x
        return x - self.dc_estimate


class Filter(Processor):
    def __init__(self, config_dict, lsl_streamer, csv_recorder): 
        super().__init__(config_dict, lsl_streamer, csv_recorder)
        self.filter_parts = []
    
    def filter(self, value):
        """
        value: a numpy array of shape (data series, channels)
        """
        for i, x in enumerate(value):
            for part in self.filter_parts:
                x = part.filter(x)
            value[i] = x
        return value


class SpindleFilter(Filter):
    def __init__(self, config_dict, lsl_streamer, csv_recorder):
        super().__init__(config_dict, lsl_streamer, csv_recorder)

        nb_channels = config_dict['nb_channels']
        sampling_rate = config_dict['frequency']
        power_line_fq = config_dict['filter_settings']['power_line']
        use_custom_fir = config_dict['filter_settings']['custom_fir']
        custom_fir_order = config_dict['filter_settings']['custom_fir_order']
        custom_fir_cutoff = config_dict['filter_settings']['custom_fir_cutoff']
        alpha_avg = config_dict['filter_settings']['polyak_mean']
        alpha_std = config_dict['filter_settings']['polyak_std']
        epsilon = config_dict['filter_settings']['epsilon']
        filter_args = config_dict['filter_settings']['filter_args']

        if len(filter_args) > 0:
            use_fir, use_notch, use_std = filter_args
        else:
            use_fir = True,
            use_notch = True,
            use_std = True
        self.use_fir = use_fir
        self.use_notch = use_notch
        self.use_std = use_std
        self.nb_channels = nb_channels
        assert power_line_fq in [50, 60], f"The only supported power line frequencies are 50 Hz and 60 Hz. Received {power_line_fq}"

        if use_custom_fir:
            fir_coef = firwin(numtaps=custom_fir_order+1, cutoff=custom_fir_cutoff, fs=sampling_rate)
        else:
            fir_coef = [
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
                0.001623780150148094927192721215192250384]

        if self.use_fir:
            self.filter_parts.append(FIR(nb_channels, fir_coef))
        if self.use_notch:
            self.filter_parts.append(Notch(power_line_fq, nb_channels))
        if self.use_std:
            self.filter_parts.append(Standardization(nb_channels, alpha_avg, alpha_std, epsilon))


class SlowOscillationFilter(Filter):
    def __init__(self, config_dict, lsl_streamer=None, csv_recorder=None):
        super().__init__(config_dict, lsl_streamer, csv_recorder)

        nb_channels = config_dict['nb_channels']
        sampling_rate = config_dict['frequency']
        power_line_fq = config_dict['filter_settings']['power_line']
        use_custom_fir = config_dict['filter_settings']['custom_fir']
        custom_fir_order = config_dict['filter_settings']['custom_fir_order']
        custom_fir_cutoff = config_dict['filter_settings']['custom_fir_cutoff']

        assert power_line_fq in [50, 60], f"The only supported power line frequencies are 50 Hz and 60 Hz. Received {power_line_fq}"

        # DC offset removal filter (high-pass filter)
        self.dc_b, self.dc_a = signal.butter(1, 0.5 / (sampling_rate / 2), "high")

        # FIR Bandpass filter (0.5 - 30 Hz)
        if use_custom_fir:
            fir_coef = firwin(numtaps=custom_fir_order+1, cutoff=custom_fir_cutoff, fs=sampling_rate)
        else:
            low = 0.5
            high = 30.0
            fir_coef = signal.firwin(20, [low, high], pass_zero=False, window="hamming", fs=sampling_rate)

        # Initialize filter states for each channel
        dc_estimate = np.zeros(nb_channels)
        alpha = 0.005

        fir = FIR(nb_channels, fir_coef)
        notch = Notch(power_line_fq, nb_channels)
        dc = DC(dc_estimate, alpha)

        self.filter_parts = [fir, notch, dc]
