import pylab,numpy 
from dateutil.parser import * #formats date strings into datetimes
from datetime import * # this allows for use of datetime objects
import math
import os # importing files
import sys # importing files
import importlib # importing libraries
from importlib import util
import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization
import statistics #Median usage
import pytz  
import pandas as pd



def ImportFileNames(csvName):

  df = pd.read_csv(csvName)
  """
  Creates the original three lists of data from the selected CSV file.  
  cgm: continuous blood glucose mmol/liter
  bas: basal in units/hour
  bol: bolus in units
  """

  old = df.old
  old = old[0:6]
  new = df.new
  
  return old, new
  
  
  # load tidals package locally if it does not exist globally
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
description: example script of how to load the tidals package
created: 2018-02-21
author: Ed Nykaza
license: BSD-2-Clause
"""

"""
Loads desired person's diabetes CSV file, and makes it accessible for data 
extraction, assuming that we have loaded in the CSVs to the google drive.

Also returns the correct size of the bins based on the cbg, basal, and bolus.
"""

def ImportData(csvName, dataType): # Good to go

  df = pd.read_csv(csvName)
  """
  Creates the original three lists of data from the selected CSV file.  
  cgm: continuous blood glucose mmol/liter
  bas: basal in units/hour
  bol: bolus in units
  """

  cgm = df.loc[df.type == "cbg", ["time", "value"]]
  bas = df.loc[df.type == "basal", ["time", "rate", "duration"]]
  bol = df.loc[df.type == "bolus", ["time", "normal"]]
      
  #Checking for what the greatest start and end bounds between glucose, basal, and bolus
  
  #glucose
  if (dataType == "new"):
    startGlucoseTime = cgm["time"][int(cgm.index[0])]
    endGlucoseTime = cgm["time"][int(cgm.index[-1])]
  else:
    endGlucoseTime = cgm["time"][int(cgm.index[0])]
    startGlucoseTime = cgm["time"][int(cgm.index[-1])]
    
  startGlucoseTime = parse(startGlucoseTime) # parses time strings into datetime values
  startGlucoseTime = startGlucoseTime.replace(microsecond = 0, tzinfo=pytz.UTC)  
  startGlucoseTime = startGlucoseTime.replace(tzinfo=None)
  
  endGlucoseTime = parse(endGlucoseTime) # parses time strings into datetime values
  endGlucoseTime = endGlucoseTime.replace(microsecond = 0, tzinfo=pytz.UTC)  
  endGlucoseTime = endGlucoseTime.replace(tzinfo=None)

  #basal
  if (dataType == "new"):
    startBasalTime = bas["time"][int(bas.index[0])]
    endBasalTime = bas["time"][int(bas.index[-1])]
  else:
    endBasalTime = bas["time"][int(bas.index[0])]
    startBasalTime = bas["time"][int(bas.index[-1])]

  startBasalTime = parse(startBasalTime) # parses time strings into datetime values
  startBasalTime = startBasalTime.replace(microsecond = 0, tzinfo=pytz.UTC)  
  startBasalTime = startBasalTime.replace(tzinfo=None) 
  
  endBasalTime = parse(endBasalTime) # parses time strings into datetime values
  endBasalTime = endBasalTime.replace(microsecond = 0, tzinfo=pytz.UTC)  
  endBasalTime = endBasalTime.replace(tzinfo=None) 
  
  #bolus
  if (dataType == "new"):
    startBolusTime = bol["time"][int(bol.index[0])]
    endBolusTime = bol["time"][int(bol.index[-1])]
  else:
    endBolusTime = bol["time"][int(bol.index[0])]
    startBolusTime = bol["time"][int(bol.index[-1])]

  startBolusTime = parse(startBolusTime) # parses time strings into datetime values
  startBolusTime = startBolusTime.replace(microsecond = 0, tzinfo=pytz.UTC)  
  startBolusTime = startBolusTime.replace(tzinfo=None) 
  
  endBolusTime = parse(endBolusTime) # parses time strings into datetime values
  endBolusTime = endBolusTime.replace(microsecond = 0, tzinfo=pytz.UTC)  
  endBolusTime = endBolusTime.replace(tzinfo=None)
  
  if (startGlucoseTime < startBasalTime):
    if (startGlucoseTime < startBolusTime):
      startBinDatetime = startGlucoseTime
    else:
      startBinDatetime = startBolusTime
  else:
    if (startBasalTime < startBolusTime):
      startBinDatetime = startBasalTime
    else:
      startBinDatetime = startBolusTime
      
  if (endGlucoseTime > endBasalTime):
    if (endGlucoseTime > endBolusTime):
      endBinDatetime = endGlucoseTime
    else:
      endBinDatetime = endBolusTime
  else:
    if (endBasalTime > endBolusTime):
      endBinDatetime = endBasalTime
    else:
      endBinDatetime = endBolusTime
  
  #standardizes start and end dates to midnight with buffer
  startBinDatetime = datetime(startBinDatetime.year, startBinDatetime.month, startBinDatetime.day, 0, 0)
  endBinDatetime = datetime(endBinDatetime.year, endBinDatetime.month, endBinDatetime.day, 0, 0) + timedelta(days = 1)
  return cgm, bas, bol, startBinDatetime, endBinDatetime
  
  
"""
This function puts the glucose data into a usable form for modeling and analysis.
We break the dates from the start of recorded data, to the end, into 5 minute 
intervals, and fill in each 'bin' with its associated blood glucose value.
We perform a weighted average by calculating the midpoint between the current 
time and its two neighbors and, assuming that the gap between said neighbors 
is less than 10 minutes, we add in the glucose value to the designated bins.
If a bin is not entirely full, then it is considered an empty bin.

@newData: type of data (as in the old data)
@tz: the time zone the person to whom these glucose bin belongs lives in
"""
  
def CreateGlucoseBins(cgm, newData, startBinDatetime, endBinDatetime):
  """
  Puts the data into a 2 dimensional list: totalBolusDayList; it contains:
    - a list of times in a special python form.
    - a list of glucose levels
  The index of the times and glucose levels match up such that each time is 
  associated with a particular glucose value.
  """
  totalGlucoseDayList = [[], []]
  for i in range(len(cgm["time"])):
    totalGlucoseDayList[0].append(cgm["time"][int(cgm.index[i])]) 
    totalGlucoseDayList[1].append(float(cgm["value"][int(cgm.index[i])]) * 18)
    
  if (newData == "old"):  
    totalGlucoseDayList[0].reverse()
    totalGlucoseDayList[1].reverse()  
    
  """
  Within the TIME list, we put the times into a more usable form, from the 
  weird python formatting, to a dateTime object that looks like:

  (year, month, day, hour, minute, second)

  This also cuts off the microseconds and time zone information
  """

  for i in range(len(totalGlucoseDayList[0])):
    totalGlucoseDayList[0][i] = parse(totalGlucoseDayList[0][i]) # parses time strings into datetime values
    totalGlucoseDayList[0][i] = totalGlucoseDayList[0][i].replace(microsecond = 0, tzinfo=pytz.UTC)  
    totalGlucoseDayList[0][i] = totalGlucoseDayList[0][i].replace(tzinfo=None) 

  
  countDatetime = startBinDatetime
  datetimeBin = []
  glucoseBinList = []

  while (countDatetime != endBinDatetime): # while the count time is not 5 mins past the last day in the sequence
    weekno = countDatetime.weekday()
    if weekno<5:
      day = "Weekday"
    else:
      day = "Weekend"
    datetimeBin = [countDatetime, None, datetime(1, 1, 1, 0, 0), day] # time at bin (lower bound), BG value of bin, portion of bin which has been filled
    glucoseBinList.append(datetimeBin)
    countDatetime = countDatetime + timedelta(minutes=5)
    

  for i in range(len(totalGlucoseDayList[0])):

    #number of seconds between now and starting date
    curDateSeconds = ((totalGlucoseDayList[0][i] - startBinDatetime).days * 24 * 60 * 60) + ((totalGlucoseDayList[0][i] - startBinDatetime).seconds)
    #timedelta only has capability to return in terms of days and seconds
    if (i == 0): #totalGlucoseDayList is in reverse chronological order, so prevDateDif is at end
      prevDateDif = 100000000 #make unreasonably large so it doesn't affect data
    else:
      prevDateSeconds = ((totalGlucoseDayList[0][i-1] - startBinDatetime).days * 24 * 60 * 60) + ((totalGlucoseDayList[0][i-1] - startBinDatetime).seconds)
      prevDateDif = curDateSeconds - prevDateSeconds
    if (i == len(totalGlucoseDayList[0]) - 1):
      nextDateDif = 100000000
    else:
      nextDateSeconds = ((totalGlucoseDayList[0][i+1] - startBinDatetime).days * 24 * 60 *60) + ((totalGlucoseDayList[0][i+1] - startBinDatetime).seconds)
      nextDateDif = nextDateSeconds - curDateSeconds
    #declare binIndex to be the bin that the current date is based in
    binIndex = int((curDateSeconds / 60) // 5) # / 60 is to convert from seconds to minutes; // 5 is to convert to 5 minute intervals
    

    if (prevDateDif <= 600): #General case (prevDate is less than 10 min away - midpoint is witin 5 min); find midpoint
      #declare midpoint
      midpoint = startBinDatetime + timedelta(seconds = prevDateSeconds + (prevDateDif/2)) #convert to datetime object for arithmetic
      
      if (midpoint >= glucoseBinList[binIndex][0]):
        #adds the weighted glucose value to the current bin in between the current time and the midpoint.
        midDateDif = (totalGlucoseDayList[0][i] - midpoint).seconds
        glucoseBinList[binIndex][2] = glucoseBinList[binIndex][2] + timedelta(seconds=midDateDif) 
        if (glucoseBinList[binIndex][1] == None):
          glucoseBinList[binIndex][1] = 0
        glucoseBinList[binIndex][1] += (midDateDif / 300) * totalGlucoseDayList[1][i]

      elif (midpoint < glucoseBinList[binIndex][0]):
        #This part adds the weighted glucose value to the current bin in between the current time and the lower edge of the bin
        prevBinDateDif = (totalGlucoseDayList[0][i] - glucoseBinList[binIndex][0]).seconds
        glucoseBinList[binIndex][2] = glucoseBinList[binIndex][2] + timedelta(seconds=prevBinDateDif) 
        if (glucoseBinList[binIndex][1] == None):
          glucoseBinList[binIndex][1] = 0
        glucoseBinList[binIndex][1] += (prevBinDateDif / 300) * totalGlucoseDayList[1][i]
        #This next part adds the weighted glucose value to the previous bin in between the upper edge of the previous bin and the midpoint
        midDateDif = (glucoseBinList[binIndex][0] - midpoint).seconds
        glucoseBinList[binIndex - 1][2] = glucoseBinList[binIndex - 1][2] + timedelta(seconds=midDateDif) 
        if (glucoseBinList[binIndex - 1][1] == None):
          glucoseBinList[binIndex - 1][1] = 0
        glucoseBinList[binIndex - 1][1] += (midDateDif / 300) * totalGlucoseDayList[1][i]

      '''
      Added filling in of the previous bin to  elif statement below:
      '''    
    elif (prevDateDif > 600): #PrevDate more than 10 min away
      #prevBinDateDif is the difference in time in seconds between the curDate's time, and the low bound on the bin's times
      prevBinDateDif = (totalGlucoseDayList[0][i] - glucoseBinList[binIndex][0]).seconds
      glucoseBinList[binIndex][2] = glucoseBinList[binIndex][2] + timedelta(seconds=prevBinDateDif) 
      if (glucoseBinList[binIndex][1] == None):
        glucoseBinList[binIndex][1] = 0
      glucoseBinList[binIndex][1] += (prevBinDateDif / 300) * totalGlucoseDayList[1][i]
      #This next part adds the weighted glucose value to the previous bin in between the upper edge of the previous bin and the midpoint
      timeInPrevBin = 300 - prevBinDateDif
      glucoseBinList[binIndex - 1][2] = glucoseBinList[binIndex - 1][2] + timedelta(seconds=timeInPrevBin) 
      if (glucoseBinList[binIndex - 1][1] == None):
        glucoseBinList[binIndex - 1][1] = 0
      glucoseBinList[binIndex - 1][1] += (timeInPrevBin / 300) * totalGlucoseDayList[1][i] 
    
    if (nextDateDif <= 600): #General case; find midpoint
      #declare midpoint
      midpoint = startBinDatetime + timedelta(seconds = curDateSeconds + (nextDateDif/2)) #convert to datetime object for arithmetic
      
      if (midpoint < glucoseBinList[binIndex + 1][0]):
        #adds the weighted glucose value to the current bin in between the current time and the midpoint.
        midDateDif = (midpoint - totalGlucoseDayList[0][i]).seconds
        glucoseBinList[binIndex][2] = glucoseBinList[binIndex][2] + timedelta(seconds=midDateDif) 
        if (glucoseBinList[binIndex][1] == None):
          glucoseBinList[binIndex][1] = 0
        glucoseBinList[binIndex][1] += (midDateDif / 300) * totalGlucoseDayList[1][i]

      elif (midpoint >= glucoseBinList[binIndex + 1][0]):
        #This part adds the weighted glucose value to the current bin in between the current time and the upper edge of the bin
        nextBinDateDif = (glucoseBinList[binIndex + 1][0] - totalGlucoseDayList[0][i]).seconds
        glucoseBinList[binIndex][2] = glucoseBinList[binIndex][2] + timedelta(seconds=nextBinDateDif)  
        if (glucoseBinList[binIndex][1] == None):
          glucoseBinList[binIndex][1] = 0
        glucoseBinList[binIndex][1] += (nextBinDateDif / 300) * totalGlucoseDayList[1][i]
        #This next part adds the weighted glucose value to the previous bin in between the lower edge of the next bin and the midpoint
        midDateDif = (midpoint - glucoseBinList[binIndex + 1][0]).seconds    
        glucoseBinList[binIndex + 1][2] = glucoseBinList[binIndex + 1][2] + timedelta(seconds=midDateDif)  
        if (glucoseBinList[binIndex + 1][1] == None):
          glucoseBinList[binIndex + 1][1] = 0
        glucoseBinList[binIndex + 1][1] += (midDateDif / 300) * totalGlucoseDayList[1][i]
    
    
      '''
      Add filling in of the next bin to  elif statement below:
     '''  
    elif (nextDateDif > 600): #NextDate is more than 10 min away
      #nextBinDateDif is the difference in time in seconds between the curDate's time, and the high bound on the bin's times
      nextBinDateDif = (glucoseBinList[binIndex + 1][0] - totalGlucoseDayList[0][i]).seconds
      glucoseBinList[binIndex][2] = glucoseBinList[binIndex][2] + timedelta(seconds=nextBinDateDif) #update the portion of bin that is filled
      if (glucoseBinList[binIndex][1] == None): #confirm there is a glucose value for this time interval
        glucoseBinList[binIndex][1] = 0
      glucoseBinList[binIndex][1] += (nextBinDateDif / 300) * totalGlucoseDayList[1][i] #update glucose value
      #This next part adds the weighted glucose value to the previous bin in between the lower edge of the next bin and the midpoint
      timeInNextBin = 300 - nextBinDateDif
      glucoseBinList[binIndex + 1][2] = glucoseBinList[binIndex + 1][2] + timedelta(seconds=timeInNextBin)  
      if (glucoseBinList[binIndex + 1][1] == None):
        glucoseBinList[binIndex + 1][1] = 0
      glucoseBinList[binIndex + 1][1] += (timeInNextBin / 300) * totalGlucoseDayList[1][i] 
      
  '''
  This section of code takes the glucoseBinList and shaves off any bins that are either 
  too big (because of some bug) or too small (not enough values).  Thus, we are left
  with only full bins, or None to represent an unfull bin.  
  '''

  '''
  Change the for loop below if you want to include "unfull" bins.
  Ex. Take the average of the surrounding bins to create a "predicted" BG value for that bin rather than setting it to 0
  '''

  for i in range(len(glucoseBinList)):
    if (glucoseBinList[i][2] < datetime(1, 1, 1, 0, 4, 55)):
      glucoseBinList[i][1] = None
      glucoseBinList[i][2] = datetime(1, 1, 1, 0, 0)

  return glucoseBinList
  
  
"""
This is the new and updated CreateBasalBins().  While the above function 
simply looks at the start time of the currrent basal amount and the end time of 
the next, this function takes the duration and applies the rate over said 
duration.  

totalBasalYearList: A 2-D list containing, for each entry, 3 values: the time it
begins, the rate in units per hour at which the insulin is added, and the 
duration the insulin lasts.

timeSlice: The amount of time from the start of the bins to the current basal
entry.

timeIndex: The time of the current bin that the basal entry should reside in.

timeSliceInBin: The amount of time of the current bin occupied by the current 
basal entry.
"""

def CreateBasalBins(bas, newData, start, end):
  totalBasalYearList = [[], [], []]
  for i in range(len(bas["time"])):
    totalBasalYearList[0].append(bas["time"][int(bas.index[i])])
    totalBasalYearList[1].append(bas["rate"][int(bas.index[i])])
    try:  
      totalBasalYearList[2].append(int(int(bas["duration"][int(bas.index[i])]) / 1000))
    except:
      totalBasalYearList[2].append(0)

  for i in range(len(totalBasalYearList[0])):
    totalBasalYearList[0][i] = parse(totalBasalYearList[0][i]) # parses time strings into datetime values
    totalBasalYearList[0][i] = totalBasalYearList[0][i].replace(microsecond = 0, tzinfo=pytz.UTC)  
    totalBasalYearList[0][i] = totalBasalYearList[0][i].replace(tzinfo=None) 
    
  for i in range(len(totalBasalYearList[1])):
    if (math.isnan(totalBasalYearList[1][i])):
      totalBasalYearList[1][i] = 0

  if (newData == "old"):
    totalBasalYearList[0].reverse()
    totalBasalYearList[1].reverse()
    totalBasalYearList[2].reverse()

  startBinDatetime = start
  endBinDatetime = end

  countDatetime = startBinDatetime
  datetimeBin = []
  basalBinList = []

  while (countDatetime != endBinDatetime): # while the count time is not 5 mins past the last day in the sequence
    weekno = countDatetime.weekday()
    if weekno < 5:
      day = "Weekday"
    else:
      day = "Weekend"
    datetimeBin = [countDatetime, 0, day] # time at bin (lower bound), BG value of bin, portion of bin which has been filled
    basalBinList.append(datetimeBin)
    countDatetime = countDatetime + timedelta(minutes=5) 

  for i in range(len(totalBasalYearList[0])):
    timeSlice = totalBasalYearList[0][i] - startBinDatetime  
    timeIndex = int(((timeSlice.days * 24 * 60 * 60) + timeSlice.seconds) / 300)
    while (totalBasalYearList[2][i] > 0):
      if (timeIndex + 1 < len(basalBinList) and (totalBasalYearList[0][i] + timedelta(seconds=totalBasalYearList[2][i])) > basalBinList[timeIndex + 1][0]):
        timeSliceInBin = (basalBinList[timeIndex + 1][0] - totalBasalYearList[0][i])
        binProportion = ((timeSliceInBin.days * 24 * 60 * 60) + timeSliceInBin.seconds) / 300
        basalBinList[timeIndex][1] += binProportion * totalBasalYearList[1][i]/12
        totalBasalYearList[0][i] = totalBasalYearList[0][i] + timedelta(seconds = (timeSliceInBin.days * 24 * 60 * 60) + timeSliceInBin.seconds)
        totalBasalYearList[2][i] = totalBasalYearList[2][i] - ((timeSliceInBin.days * 24 * 60 * 60) + timeSliceInBin.seconds) 
        timeIndex += 1
      else:
        binProportion = totalBasalYearList[2][i] / 300
        basalBinList[timeIndex][1] += binProportion * totalBasalYearList[1][i]/12
        totalBasalYearList[0][i] = totalBasalYearList[0][i] + timedelta(seconds = totalBasalYearList[2][i])
        totalBasalYearList[2][i] = totalBasalYearList[2][i] - totalBasalYearList[2][i]  
      
  return basalBinList
  
  
"""
This function creates the bins containing the bolus values.  Because bolus is a 
single infusion of insulin, however, we put the total "normal" amount into the 
bin that bounds the time of infusion.
"""
  
def CreateBolusBins(bol, newData, start, end):
  totalBolusYearList = [[], []]
  for i in range(len(bol["time"])):
    totalBolusYearList[0].append(bol["time"][int(bol.index[i])])
    totalBolusYearList[1].append(bol["normal"][int(bol.index[i])])

  for i in range(len(totalBolusYearList[0])):
    totalBolusYearList[0][i] = parse(totalBolusYearList[0][i]) # parses time strings into datetime values
    totalBolusYearList[0][i] = totalBolusYearList[0][i].replace(microsecond = 0, tzinfo=pytz.UTC)  
    totalBolusYearList[0][i] = totalBolusYearList[0][i].replace(tzinfo=None) 
  
  if (newData != "new"):
    totalBolusYearList[0].reverse()
    totalBolusYearList[1].reverse()  

  startBinDatetime = start  
  endBinDatetime = end

  countDatetime = startBinDatetime
  datetimeBin = []
  bolusBinList = []

  while (countDatetime != endBinDatetime): # while the count time is not 5 mins past the last day in the sequence
    weekno = countDatetime.weekday()
    if weekno < 5:
      day = "Weekday"
    else:
      day = "Weekend"
    datetimeBin = [countDatetime, 0, day] # time at bin (lower bound), BG value of bin, portion of bin which has been filled
    bolusBinList.append(datetimeBin)
    countDatetime = countDatetime + timedelta(minutes=5) 

  for i in range(len(totalBolusYearList[0])):
    timeSlice = totalBolusYearList[0][i] - startBinDatetime  
    timeIndex = int(((timeSlice.days * 24 * 60 * 60) + timeSlice.seconds) / 300) 
    bolusBinList[timeIndex][1] += totalBolusYearList[1][i]

          
  return bolusBinList
