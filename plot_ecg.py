#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmullie

Code from dicom-ecg-plot - The MIT License (MIT),
copyright (c) 2013 Marco De Benedetto <debe@galliera.it>

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import struct
from scipy.signal import butter, lfilter

LAYOUT = {'3x4_1': [[0, 3, 6, 9],
                     [1, 4, 7, 10],
                     [2, 5, 8, 11],
                     [1]],
              '3x4': [[0, 3, 6, 9],
                      [1, 4, 7, 10],
                      [2, 5, 8, 11]],
              '6x2': [[0, 6],
                      [1, 7],
                      [3, 8],
                      [4, 9],
                      [5, 10],
                      [6, 11]],
              '12x1': [[0],
                       [1],
                       [2],
                       [3],
                       [4],
                       [5],
                       [6],
                       [7],
                       [8],
                       [9],
                       [10],
                       [11]]}
                       

def butter_lowpass(highcut, sampfreq, order):
  nyquist_freq = .5 * sampfreq
  high = highcut / nyquist_freq
  num, denom = butter(order, high, btype='lowpass')
  return num, denom
    
def butter_lowpass_filter(data, highcut, sampfreq, order):
  num, denom = butter_lowpass(highcut, sampfreq, order=order)
  return lfilter(num, denom, data)
    
def get_signals(waveform_data, num_channels, channel_definitions, 
  samples, sampling_frequency):
  
  factor = np.zeros(num_channels) + 1
  baseln = np.zeros(num_channels)
  units = []
  for idx in range(num_channels):
    definition = channel_definitions[idx]
        
    assert (definition.WaveformBitsStored == 16)
        
    if definition.get('ChannelSensitivity'):
        factor[idx] = (
            float(definition.ChannelSensitivity) *
            float(definition.ChannelSensitivityCorrectionFactor)
        )
        
    if definition.get('ChannelBaseline'):
        baseln[idx] = float(definition.get('ChannelBaseline'))
        
    units.append(
        definition.ChannelSensitivityUnitsSequence[0].CodeValue
    )
          
  unpack_fmt = '<%dh' % (len(waveform_data) / 2)
  unpacked_waveform_data = struct.unpack(unpack_fmt, waveform_data)
  signals = np.asarray(
    unpacked_waveform_data,
    dtype=np.float32).reshape(
    samples,
    num_channels).transpose()
          
  for channel in range(num_channels):
    signals[channel] = (
        (signals[channel] + baseln[channel]) * factor[channel]
    )
        
  high = 40.0
          
  # conversion factor to obtain millivolts values
  millivolts = {'uV': 1000.0, 'mV': 1.0}
          
  for i, signal in enumerate(signals):
    signals[i] = butter_lowpass_filter(
        np.asarray(signal),
        high,
        sampling_frequency,
        order=2
    ) / millivolts[units[i]]
          
  return signals

def create_figure(left, right, top, bottom, height, samples):
  # Init figure and axes
  fig = plt.figure(tight_layout=False)
  axes = fig.add_subplot(1, 1, 1)
  
  fig.subplots_adjust(left=left, right=right, top=top,
                      bottom=bottom)
  
  axes.set_ylim([0, height])
  
  # We want to plot N points, where N=number of samples
  axes.set_xlim([0, samples - 1])
  return fig, axes

def plot(signals, channel_definitions, samples, layoutid, mm_mv=10.0):
  
  paper_w, paper_h = 297.0, 210.0
  
  width = 250.0
  height = 170.0
  margin_left = margin_right = .5 * (paper_w - width)
  margin_bottom = 10.0
  
  # Normalized in [0, 1]
  left = margin_left / paper_w
  right = left + width / paper_w
  bottom = margin_bottom / paper_h
  top = bottom + height / paper_h
      
  fig, axis = create_figure(left, right, top, bottom, height, samples)
      
  layout = LAYOUT[layoutid]
  rows = len(layout)
      
  for numrow, row in enumerate(layout):
      
      columns = len(row)
      row_height = height / rows
      
      # Horizontal shift for lead labels and separators
      h_delta = samples / columns
      
      # Vertical shift of the origin
      v_delta = round(
          height * (1.0 - 1.0 / (rows * 2)) -
          numrow * (height / rows)
      )
      
      # Let's shift the origin on a multiple of 5 mm
      v_delta = (v_delta + 2.5) - (v_delta + 2.5) % 5
      
      # Lenght of a signal chunk
      chunk_size = int(samples / len(row))
      for numcol, signum in enumerate(row):
          left = numcol * chunk_size
          right = (1 + numcol) * chunk_size
      
          # The signal chunk, vertical shifted and
          # scaled by mm/mV factor
          signal = v_delta + mm_mv * signals[signum][left:right]
          axis.plot(
              list(range(left, right)),
              signal,
              clip_on=False,
              linewidth=0.6,
              color='black',
              zorder=10)
      
          cseq = channel_definitions[signum].ChannelSourceSequence
          meaning = cseq[0].CodeMeaning.replace(
              'Lead', '').replace('(Einthoven)', '')
      
          h = h_delta * numcol
          v = v_delta + row_height / 2.6
          plt.plot(
              [h, h],
              [v - 3, v],
              lw=.6,
              color='black',
              zorder=50
          )
      
          axis.text(
              h + 40,
              v_delta + row_height / 3,
              meaning,
              zorder=50,
              fontsize=8
          )
      
  # A4 size in inches
  fig.set_size_inches(11.69, 8.27)
        
dicom = read_dcm('./ecg.dcm')
waveform_sequence = dicom.get('WaveformSequence')
sequence_length = len(waveform_sequence)

for ecg in waveform_sequence:
  
  waveform_data = ecg.get('WaveformData')
  num_channels = ecg.get('NumberOfWaveformChannels')
  channel_definitions = ecg.get('ChannelDefinitionSequence')
  samples = ecg.get('NumberOfWaveformSamples')
  sampling_frequency = ecg.get('SamplingFrequency')
  duration = samples / sampling_frequency
  
  signals = get_signals(waveform_data, num_channels, 
    channel_definitions, samples, sampling_frequency)
  
  for channel in channel_definitions:
    source_sequence = channel.get('ChannelSourceSequence')
    lead_name = None
    for item in source_sequence:
      lead_name = item.get('CodeMeaning')
      if lead_name: break
    print(lead_name)
  
  print(waveform_data)
  
  plot(signals, channel_definitions, samples, '3x4')
  
  plt.show()


#with h5py.File('./ecg.h5', 'w') as data_file:
#  data_file.create_dataset('dicom', data=pixel_array)