ADS_GAIN = 24
ADS_LSB = (2 / ADS_GAIN) / (2**24 - 1)  # needs to be multiplied by VREF

DEFAULT_FRONTEND_CONFIG = [
    # nomenclature: name [default setting] [bits 7-0] : description
    # Read only ID:
    0x3E, # ID [xx] [REV_ID[2:0], 1, DEV_ID[1:0], NU_CH[1:0]] : (RO)
    # Global Settings Across Channels:
    0x96, # CONFIG1 [96] [1, DAISY_EN(bar), CLK_EN, 1, 0, DR[2:0]] : Datarate = 250 SPS
    0xC0, # CONFIG2 [C0] [1, 1, 0, INT_CAL, 0, CAL_AMP0, CAL_FREQ[1:0]] : No tests
    0x60, # CONFIG3 [60] [PD_REFBUF(bar), 1, 1, BIAS_MEAS, BIASREF_INT, PD_BIAS(bar), BIAS_LOFF_SENS, BIAS_STAT] : Power-down reference buffer, no bias
    0x00, # LOFF [00] [COMP_TH[2:0], 0, ILEAD_OFF[1:0], FLEAD_OFF[1:0]] : No lead-off
    # Channel-Specific Settings:
    0x61, # CH1SET [61] [PD1, GAIN1[2:0], SRB2, MUX1[2:0]] : Channel 1 active, 24 gain, no SRB2 & input shorted
    0x61, # CH2SET [61] [PD2, GAIN2[2:0], SRB2, MUX2[2:0]] : Channel 2 active, 24 gain, no SRB2 & input shorted
    0x61, # CH3SET [61] [PD3, GAIN3[2:0], SRB2, MUX3[2:0]] : Channel 3 active, 24 gain, no SRB2 & input shorted
    0x61, # CH4SET [61] [PD4, GAIN4[2:0], SRB2, MUX4[2:0]] : Channel 4 active, 24 gain, no SRB2 & input shorted
    0x61, # CH5SET [61] [PD5, GAIN5[2:0], SRB2, MUX5[2:0]] : Channel 5 active, 24 gain, no SRB2 & input shorted
    0x61, # CH6SET [61] [PD6, GAIN6[2:0], SRB2, MUX6[2:0]] : Channel 6 active, 24 gain, no SRB2 & input shorted
    0x61, # CH7SET [61] [PD7, GAIN7[2:0], SRB2, MUX7[2:0]] : Channel 7 active, 24 gain, no SRB2 & input shorted
    0x61, # CH8SET [61] [PD8, GAIN8[2:0], SRB2, MUX8[2:0]] : Channel 8 active, 24 gain, no SRB2 & input shorted
    0x00, # BIAS_SENSP [00] [BIASP8, BIASP7, BIASP6, BIASP5, BIASP4, BIASP3, BIASP2, BIASP1] : No bias
    0x00, # BIAS_SENSN [00] [BIASN8, BIASN7, BIASN6, BIASN5, BIASN4, BIASN3, BIASN2, BIASN1] No bias
    0x00, # LOFF_SENSP [00] [LOFFP8, LOFFP7, LOFFP6, LOFFP5, LOFFP4, LOFFP3, LOFFP2, LOFFP1] : No lead-off
    0x00, # LOFF_SENSN [00] [LOFFM8, LOFFM7, LOFFM6, LOFFM5, LOFFM4, LOFFM3, LOFFM2, LOFFM1] : No lead-off
    0x00, # LOFF_FLIP [00] [LOFF_FLIP8, LOFF_FLIP7, LOFF_FLIP6, LOFF_FLIP5, LOFF_FLIP4, LOFF_FLIP3, LOFF_FLIP2, LOFF_FLIP1] : No lead-off flip
    # Lead-Off Status Registers (Read-Only Registers):
    0x00, # LOFF_STATP [00] [IN8P_OFF, IN7P_OFF, IN6P_OFF, IN5P_OFF, IN4P_OFF, IN3P_OFF, IN2P_OFF, IN1P_OFF] : Lead-off positive status (RO)
    0x00, # LOFF_STATN [00] [IN8M_OFF, IN7M_OFF, IN6M_OFF, IN5M_OFF, IN4M_OFF, IN3M_OFF, IN2M_OFF, IN1M_OFF] : Laed-off negative status (RO)
    # GPIO and OTHER Registers:
    0x0F, # GPIO [0F] [GPIOD[4:1], GPIOC[4:1]] : All GPIOs as inputs
    0x00, # MISC1 [00] [0, 0, SRB1, 0, 0, 0, 0, 0] : Disable SRBM
    0x00, # MISC2 [00] [00] : Unused
    0x00, # CONFIG4 [00] [0, 0, 0, 0, SINGLE_SHOT, 0, PD_LOFF_COMP(bar), 0] : Single-shot, lead-off comparator disabled
]

FRONTEND_CONFIG = [
    0x3E, # ID (RO)
    0x95, # CONFIG1 [95] [1, DAISY_EN(bar), CLK_EN, 1, 0, DR[2:0]] : Datarate = 500 SPS
    0xD0, # CONFIG2 [C0] [1, 1, 0, INT_CAL, 0, CAL_AMP0, CAL_FREQ[1:0]]
    0xFC, # CONFIG3 [E0] [PD_REFBUF(bar), 1, 1, BIAS_MEAS, BIASREF_INT, PD_BIAS(bar), BIAS_LOFF_SENS, BIAS_STAT] : Power-down reference buffer, no bias
    0x00, # No lead-off
    0x62, # CH1SET [60] [PD1, GAIN1[2:0], SRB2, MUX1[2:0]] set to measure BIAS signal
    0x60, # CH2SET
    0x60, # CH3SET
    0x60, # CH4SET
    0x60, # CH5SET
    0x60, # CH6SET
    0x60, # CH7SET
    0x60, # CH8SET
    0x00, # BIAS_SENSP 00
    0x00, # BIAS_SENSN 00
    0x00, # LOFF_SENSP Lead-off on all positive pins?
    0x00, # LOFF_SENSN Lead-off on all negative pins?
    0x00, # Normal lead-off
    0x00, # Lead-off positive status (RO)
    0x00, # Lead-off negative status (RO)
    0x00, # All GPIOs as output ?
    0x20, # Enable SRB1
]


LEADOFF_CONFIG = [
    0x3E, # ID (RO)
    0x95, # CONFIG1 [95] [1, DAISY_EN(bar), CLK_EN, 1, 0, DR[2:0]] : Datarate = 500 SPS
    0xC0, # CONFIG2 [C0] [1, 1, 0, INT_CAL, 0, CAL_AMP0, CAL_FREQ[1:0]]
    0xFC, # CONFIG3 [E0] [PD_REFBUF(bar), 1, 1, BIAS_MEAS, BIASREF_INT, PD_BIAS(bar), BIAS_LOFF_SENS, BIAS_STAT] : Power-down reference buffer, no bias
    0x00, # No lead-off
    0x60, # CH1SET [60] [PD1, GAIN1[2:0], SRB2, MUX1[2:0]] set to measure BIAS signal
    0x60, # CH2SET
    0x60, # CH3SET
    0x60, # CH4SET
    0x60, # CH5SET
    0x60, # CH6SET
    0x60, # CH7SET
    0x60, # CH8SET
    0x00, # BIAS_SENSP 00
    0x00, # BIAS_SENSN 00
    0xFF, # LOFF_SENSP Lead-off on all positive pins?
    0xFF, # LOFF_SENSN Lead-off on all negative pins?
    0x00, # Normal lead-off
    0x00, # Lead-off positive status (RO)
    0x00, # Lead-off negative status (RO)
    0x00, # All GPIOs as output ?
    0x20, # Enable SRB1
    0x00,
    0x02,
]

def to_ads_frequency(frequency):
    possible_datarates = [250, 500, 1000, 2000, 4000, 8000, 16000]
    dr = 16000
    for i in possible_datarates:
        if i >= frequency:
            dr = i
            break
    return dr
    
def mod_config(config, datarate, channel_modes):
    
    # datarate:
    possible_datarates = [(250, 0x06),
                          (500, 0x05),
                          (1000, 0x04),
                          (2000, 0x03),
                          (4000, 0x02),
                          (8000, 0x01),
                          (16000, 0x00)]
    mod_dr = 0x00
    for i, j in possible_datarates:
        if i >= datarate:
            mod_dr = j
            break
    
    new_cf1 = config[1] & 0xF8
    new_cf1 = new_cf1 | mod_dr
    config[1] = new_cf1
    
    # bias:
    # assert len(channel_modes) == 7
    config[13] = 0x00  # clear BIAS_SENSP
    config[14] = 0x00  # clear BIAS_SENSN
    for chan_i, chan_mode in enumerate(channel_modes):
        config_idx = chan_i + 5

        mod = config[config_idx] & 0x78  # clear PDn and MUX[2:0]
        if chan_mode == 'simple':
            # If channel is activated, we send the channel's output to the BIAS mechanism
            bit_i = 1 << chan_i
            config[13] = config[13] | bit_i
            config[14] = config[14] | bit_i
        elif chan_mode == 'bias':
            # Set the last 3 bits to 010 to use for BIAS measurement
            mod = mod | 0x02
        elif chan_mode == 'disabled':
            mod = mod | 0x81  # PDn = 1 and input shorted (001)
        else:
            assert False, f"Wrong key: {chan_mode}."
        config[config_idx] = mod
    # for n, c in enumerate(config):  # print ADS1299 configuration registers
    #     print(f"config[{n}]:\t{c:08b}\t({hex(c)})")
    return config