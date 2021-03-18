#include <iostream>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include<bits/stdc++.h>
#include <algorithm>

using namespace std;

bool sortcol(const vector<float>& v1,const vector<float>& v2) { 
    return v1[0] < v2[0]; 
} 

bool sortmap(const unordered_map<string, float>& v1,const unordered_map<string, float>& v2) { 
    return v1.at("onset_time") < v2.at("onset_time"); 
} 

class RegressionPostProcessing {
    private:
        const int framesPerSecond = 100;
        const int classesNum = 88;
        const float onsetThreshold = 0.3;
        const float offsetThreshold = 0.3;
        const float frameThreshold = 0.1;
        const float pedalOffset_threshold = 0.2;
        const int beginNote = 21;
        const int velocityScale = 128;
    public:
        RegressionPostProcessing();
        unordered_map<string, vector<unordered_map<string, float>>>   outputMapToMidiEvents(unordered_map<string, vector<vector<float>>>);

        unordered_map<string, vector<vector<float>>>    outputMapToNotePedalArrays(unordered_map<string, vector<vector<float>>> );

        unordered_map<string, vector<vector<float>>>    getBinarizedOutputFromRegression(vector<vector<float>>, float, int);

        vector<vector<float>>   outputMapToDetectedNotes(unordered_map<string, vector<vector<float>>>);

        vector<vector<float>>   outputMapToDetectedPedals(unordered_map<string, vector<vector<float>>>);

        vector<unordered_map<string, float>>     detectedNotesToEvents(vector<vector<float>>);

        vector<unordered_map<string, float>>     detectedPedalsToEvents(vector<vector<float>>);

        bool isMonotonicNeighbour(vector<float>, int, int);

        vector<vector<float>> zerosLike(vector<vector<float>>);

        vector<float> slicing(vector<vector<float>>, int);

        vector<vector<float>> noteDetectionWithOnsetOffsetRegress(
            vector<float> frameOutput, vector<float> onsetOutput,
            vector<float> onsetShiftOutput, vector<float> offsetOutput,
            vector<float> offsetShiftOutput, vector<float> velocityOutput,
            float frameThreshold
        );

        vector<vector<float>> pedalDetectionWithOnsetOffsetRegress(
            vector<float> frameOutput, vector<float> offsetOutput,
            vector<float> offsetShiftOutput, float frameThreshold
        );

        vector<float> concatVectors(int, int);
        vector<float> concatVectors(vector<float>, int);
        vector<float> concatVectors(vector<float>, vector<float>);
        vector<vector<float>> concatVectors(vector<vector<float>>, vector<vector<float>>);

        vector<float> divideBy(vector<float>, int);

        vector<vector<float>> stack(vector<float>, vector<float>, vector<float>, vector<float>);
        vector<vector<float>> stack(vector<float>, vector<float>);
};

RegressionPostProcessing::RegressionPostProcessing(){

}

unordered_map<string, vector<unordered_map<string, float>>> RegressionPostProcessing::outputMapToMidiEvents(unordered_map<string, vector<vector<float>>> outputMap) {
    /*
    * Main function. Post process model outputs to MIDI events.

    *    Args:
        *  output_dict: {
        *    'reg_onset_output': (segment_frames, classes_num), 
        *    'reg_offset_output': (segment_frames, classes_num), 
        *    'frame_output': (segment_frames, classes_num), 
        *    'velocity_output': (segment_frames, classes_num), 
        *    'reg_pedal_onset_output': (segment_frames, 1), 
        *    'reg_pedal_offset_output': (segment_frames, 1), 
        *    'pedal_frame_output': (segment_frames, 1)}

    *    Outputs:
        *  est_note_events: list of dict, e.g. [
        *    {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83}, 
        *    {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]
        *  est_pedal_events: list of dict, e.g. [
        *    {'onset_time': 0.17, 'offset_time': 0.96}, 
        *    {'osnet_time': 1.17, 'offset_time': 2.65}]
    */

    // Post process piano note outputs to piano note and pedal events information
    unordered_map<string, vector<vector<float>>> estOnOffNoteVelsAndPedals = this->outputMapToNotePedalArrays(outputMap);
    vector<vector<float>> est_on_off_note_vels = estOnOffNoteVelsAndPedals["est_on_off_note_vels"];

    /*  
    * est_on_off_note_vels: (events_num, 4), the four columns are: [onset_time, offset_time, piano_note, velocity], 
    * est_pedal_on_offs: (pedal_events_num, 2), the two columns are: [onset_time, offset_time]
    */

    // Reformat notes to MIDI events
    vector<unordered_map<string, float>> estNoteEvents = this->detectedNotesToEvents(estOnOffNoteVelsAndPedals["est_on_off_note_vels"]);
    vector<unordered_map<string, float>> estPedalEvents;
    if (estOnOffNoteVelsAndPedals["est_on_off_note_vels"].size() > 0) {
        estPedalEvents = this->detectedPedalsToEvents(estOnOffNoteVelsAndPedals["est_pedal_events"]);
    } else {
        estPedalEvents = {};
    }

    sort(estNoteEvents.begin(), estNoteEvents.end(), sortmap);
    sort(estPedalEvents.begin(), estPedalEvents.end(), sortmap);
    int i = 0;
    while (i < estNoteEvents.size()) {
        if (estNoteEvents[i].at("velocity") > 1.28 || estNoteEvents[i].at("velocity") <= 0) {
            estNoteEvents.erase(estNoteEvents.begin() + i);
        } else {
            i++;
        }
    }

    unordered_map<string, vector<unordered_map<string, float>>> output;

    output["est_note_events"] = estNoteEvents;
    output["est_pedal_events"] = estPedalEvents;

    return output;
}

unordered_map<string, vector<vector<float>>> RegressionPostProcessing::outputMapToNotePedalArrays(unordered_map<string, vector<vector<float>>> outputMap) {
    /* 
    * Postprocess the output probabilities of a transription model to MIDI 
    *    events.

    *    Args:
    *      output_dict: dict, {
    *        'reg_onset_output': (frames_num, classes_num), 
    *        'reg_offset_output': (frames_num, classes_num), 
    *        'frame_output': (frames_num, classes_num), 
    *        'velocity_output': (frames_num, classes_num), 
    *        ...}

    *    Returns:
    *      est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time, 
    *        offset_time, piano_note and velocity. E.g. [
    *         [39.74, 39.87, 27, 0.65], 
    *         [11.98, 12.11, 33, 0.69], 
    *         ...]

    *      est_pedal_on_offs: (pedal_events_num, 2), the 2 columns are onset_time 
    *        and offset_time. E.g. [
    *         [0.17, 0.96], 
    *         [1.17, 2.65], 
    *         ...]
    * */

    // ------ 1. Process regression outputs to binarized outputs ------
    // For example, onset or offset of [0., 0., 0.15, 0.30, 0.40, 0.35, 0.20, 0.05, 0., 0.]
    // will be processed to [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]

    // Calculate binarized onset output from regression output
    unordered_map<string, vector<vector<float>>> onsetBinarOutput = this->getBinarizedOutputFromRegression(
                                        outputMap["reg_onset_output"], this->onsetThreshold, 2);

    outputMap["onset_output"] = onsetBinarOutput["output"]; // Values are 0 or 1
    outputMap["onset_shift_output"] = onsetBinarOutput["shift_output"];

    unordered_map<string, vector<vector<float>>> offsetBinarOutput = this->getBinarizedOutputFromRegression(
                                        outputMap["reg_offset_output"], this->offsetThreshold, 4);

    outputMap["offset_output"] = offsetBinarOutput["output"]; // Values are 0 or 1
    outputMap["offset_shift_output"] = offsetBinarOutput["shift_output"];

    if (outputMap.find("reg_pedal_onset_output") != outputMap.end()) {
        /*
        *  Pedal onsets are not used in inference. Instead, frame-wise pedal
        *  predictions are used to detect onsets. We empirically found this is 
        *  more accurate to detect pedal onsets.
        */
    }

    if (outputMap.find("reg_pedal_offset_output") != outputMap.end()) {
        // Calculate binarized pedal offset output from regression output
        unordered_map<string, vector<vector<float>>> offsetPedalBinarOutput = this->getBinarizedOutputFromRegression(
                                        outputMap["reg_pedal_offset_output"], this->offsetThreshold, 4);

        outputMap["pedal_offset_output"] = offsetPedalBinarOutput["output"]; // Values are 0 or 1;
        outputMap["pedal_offset_shift_output"] = offsetPedalBinarOutput["shift_output"];
    }

    // ------ 2. Process matrices results to event results ------
    // Detect piano notes from output_dict
    vector<vector<float>> estOnOffNoteVels = this->outputMapToDetectedNotes(outputMap);

    vector<vector<float>> estPedalOnOffs;
    if  (outputMap.find("reg_pedal_onset_output") != outputMap.end()) {
        // Detect piano pedals from output_dict
        estPedalOnOffs = this->outputMapToDetectedPedals(outputMap);
    }

    unordered_map<string, vector<vector<float>>> returnValue;
    returnValue["est_on_off_note_vels"] = estOnOffNoteVels;
    returnValue["est_pedal_on_offs"] = estPedalOnOffs;

    return returnValue;
}

unordered_map<string,vector<vector<float>>> RegressionPostProcessing::getBinarizedOutputFromRegression(vector<vector<float>> regOutput, float threshold, int neighbour) {
    /*
    * Calculate binarized output and shifts of onsets or offsets from the
    *    regression results.

    *    Args:
    *      reg_output: (frames_num, classes_num)
    *      threshold: float
    *      neighbour: int

    *    Returns:
    *      binary_output: (frames_num, classes_num)
    *      shift_output: (frames_num, classes_num)
     */

    vector<vector<float>> binaryOutput = this->zerosLike(regOutput);
    vector<vector<float>> shiftOutput = this->zerosLike(regOutput);

    int framesNum = regOutput.size();
    int classesNum = regOutput[0].size();
    
    for (int i = 0; i < classesNum; i++)
    {   
        vector<float> x = this->slicing(regOutput,i);
        for (int n = neighbour; n < framesNum-neighbour; n++){
            if (x[n] > threshold && this->isMonotonicNeighbour(x, n, neighbour)){
                binaryOutput[n][i] = 1;
                /*
                *See Section III-D in [1] for deduction.
                *    [1] Q. Kong, et al., High-resolution Piano Transcription 
                *    with Pedals by Regressing Onsets and Offsets Times, 2020.
                */
                float shift;
                if (x[n-1] > x[n+1]) {
                    shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2;
                } else {
                    shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2;
                }

                shiftOutput[n][i] = shift;
            }
        }
    }

    unordered_map<string, vector<vector<float>>> out;
    out["output"] = binaryOutput;
    out["shift_output"] = shiftOutput;
    
    return out;
}

vector<vector<float>>  RegressionPostProcessing::outputMapToDetectedNotes(unordered_map<string, vector<vector<float>>> outputMap) {
    /*
    * Postprocess output_dict to piano notes.

    *    Args:
    *      output_dict: dict, e.g. {
    *        'onset_output': (frames_num, classes_num),
    *        'onset_shift_output': (frames_num, classes_num),
    *        'offset_output': (frames_num, classes_num),
    *        'offset_shift_output': (frames_num, classes_num),
    *        'frame_output': (frames_num, classes_num),
    *        'onset_output': (frames_num, classes_num),
    *        ...}

    *    Returns:
    *      est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets, 
    *      MIDI notes and velocities. E.g.,
    *        [[39.7375, 39.7500, 27., 0.6638],
    *         [11.9824, 12.5000, 33., 0.6892],
    *         ...]
    */

    vector<vector<float>> estTuples;
    vector<float> estMidiNotes;
    int classesNum = outputMap["frame_output"][0].size();

    for (int pianoNote = 0; pianoNote < classesNum; pianoNote++)
    {
        vector<vector<float>> estTuplesPerNote = this->noteDetectionWithOnsetOffsetRegress(
            this->slicing(outputMap["frame_output"], pianoNote), 
            this->slicing(outputMap["onset_output"], pianoNote), 
            this->slicing(outputMap["onset_shift_output"], pianoNote), 
            this->slicing(outputMap["offset_output"], pianoNote), 
            this->slicing(outputMap["offset_shift_output"], pianoNote), 
            this->slicing(outputMap["velocity_output"], pianoNote), 
            this->frameThreshold
        );
        estTuples = this->concatVectors(estTuples, estTuplesPerNote);
        estMidiNotes = this->concatVectors(estMidiNotes, this->concatVectors(pianoNote + this->beginNote, estTuplesPerNote.size()));
    }
    
    vector<float> onsetTimes = this->divideBy(this->concatVectors(this->slicing(estTuples, 0), this->slicing(estTuples, 2)), this->framesPerSecond);
    vector<float> offsetTimes = this->divideBy(this->concatVectors(this->slicing(estTuples, 1), this->slicing(estTuples, 3)), this->framesPerSecond);
    vector<float> velocity = this->slicing(estTuples, 4);

    vector<vector<float>> estOnOffNoteVels = this->stack(onsetTimes, offsetTimes, estMidiNotes, velocity);

    return estOnOffNoteVels;
}

vector<vector<float>>  RegressionPostProcessing::outputMapToDetectedPedals(unordered_map<string, vector<vector<float>>> outputMap) {
    /*
    * Postprocess output_dict to piano pedals.

    *    Args:
    *      output_dict: dict, e.g. {
    *        'pedal_frame_output': (frames_num,),
    *        'pedal_offset_output': (frames_num,),
    *        'pedal_offset_shift_output': (frames_num,),
    *        ...}

    *    Returns:
    *      est_on_off: (notes, 2), the two columns are pedal onsets and pedal
    *        offsets. E.g.,
    *          [[0.1800, 0.9669],
    *           [1.1400, 2.6458],
    *           ...]
    */

    int framesNum = outputMap["pedal_frame_output"][0].size();

    vector<vector<float>> estTuples = this->pedalDetectionWithOnsetOffsetRegress(
        this->slicing(outputMap["pedal_frame_output"], 0),
        this->slicing(outputMap["pedal_offset_output"], 0),
        this->slicing(outputMap["pedal_offset_shift_output"], 0), 0.5
    );

    if (estTuples.size() == 0) {
        return estTuples;
    } else {
        vector<float> onsetTimes = this->divideBy(this->concatVectors(this->slicing(estTuples, 0), this->slicing(estTuples, 2)), this->framesPerSecond);
        vector<float> offsetTimes = this->divideBy(this->concatVectors(this->slicing(estTuples, 1), this->slicing(estTuples, 3)), this->framesPerSecond);

        vector<vector<float>> estOnOff = this->stack(onsetTimes, offsetTimes);
        return estOnOff;
    }
}

vector<unordered_map<string, float>> RegressionPostProcessing::detectedNotesToEvents(vector<vector<float>> estOnOffNoteVels) {
    /*
    * Reformat detected notes to midi events.

    *    Args:
    *      est_on_off_vels: (notes, 3), the three columns are onset_times, 
    *        offset_times and velocity. E.g.
    *        [[32.8376, 35.7700, 0.7932],
    *         [37.3712, 39.9300, 0.8058],
    *         ...]
        
    *    Returns:
    *      midi_events, list, e.g.,
    *        [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
    *         {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
    *         ...]
    */

    vector<unordered_map<string, float>>  midiEvents;

    for (vector<float> tmp : estOnOffNoteVels) {
        unordered_map<string, float> temp;
        temp["onset_time"] = tmp[0];
        temp["offset_time"] = tmp[1];
        temp["midi_note"] = tmp[2];
        temp["velocity"] = tmp[3];
        midiEvents.push_back(temp);
    }

    return midiEvents;
}

vector<unordered_map<string, float>> RegressionPostProcessing::detectedPedalsToEvents(vector<vector<float>> estOnOffNoteVels) {
    /*
    * Reformat detected pedal onset and offsets to events.

    *    Args:
    *      pedal_on_offs: (notes, 2), the two columns are pedal onsets and pedal
    *      offsets. E.g., 
    *        [[0.1800, 0.9669],
    *         [1.1400, 2.6458],
    *         ...]

    *    Returns:
    *      pedal_events: list of dict, e.g.,
    *        [{'onset_time': 0.1800, 'offset_time': 0.9669}, 
    *         {'onset_time': 1.1400, 'offset_time': 2.6458},
    *         ...]
    */
    vector<unordered_map<string, float>>  midiEvents;

    for (vector<float> tmp : estOnOffNoteVels) {
        unordered_map<string, float> temp;
        temp["onset_time"] = tmp[0];
        temp["offset_time"] = tmp[1];
        midiEvents.push_back(temp);
    }

    return midiEvents;
}

vector<vector<float>> RegressionPostProcessing::noteDetectionWithOnsetOffsetRegress(
    vector<float> frameOutput, vector<float> onsetOutput,
            vector<float> onsetShiftOutput, vector<float> offsetOutput,
            vector<float> offsetShiftOutput, vector<float> velocityOutput,
            float frameThreshold
) {
    /*
    *    Process prediction matrices to note events information.
    *    First, detect onsets with onset outputs. Then, detect offsets
    *    with frame and offset outputs.

    *    Args:
    *    frame_output: (frames_num,)
    *    onset_output: (frames_num,)
    *    onset_shift_output: (frames_num,)
    *    offset_output: (frames_num,)
    *    offset_shift_output: (frames_num,)
    *    velocity_output: (frames_num,)
    *    frame_threshold: float

    *    Returns: 
    *    output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity], 
    *    e.g., [
    *        [1821, 1909, 0.47498, 0.3048533, 0.72119445], 
    *        [1909, 1947, 0.30730522, -0.45764327, 0.64200014], 
    *        ...]
    */
    vector<vector<float>> outputTuples;
    int bgn;
    int frameDisappear;
    int offsetOccur;
    int fin;

    for (int i = 0; i < onsetOutput.size(); i++) {
        if (onsetOutput[i] == 1) {
            // Onset detected
            if (bgn) {
                // Consecutive onsets. E.g., pedal is not released, but two 
                // consecutive notes being played.
                fin = i - 1;
                vector<float> tmp;
                tmp.push_back(bgn);
                tmp.push_back(fin);
                tmp.push_back(onsetShiftOutput[bgn]);
                tmp.push_back(0);
                tmp.push_back(velocityOutput[bgn]);
                outputTuples.push_back(tmp);

                frameDisappear = 0;
                offsetOccur = 0;
            }
            bgn = i;
        }

        if (bgn && bgn < i) {
            // If onset found, then search offset

            if (frameOutput[i] <= frameThreshold && !frameDisappear) {
                // Frame disappear detected
                frameDisappear = i;
            }
            if (offsetOutput[i] == 1 && !offsetOccur){
                // Offset detected
                offsetOccur = i;
            }
            if (frameDisappear) {
                if (offsetOccur && offsetOccur - bgn > frameDisappear - offsetOccur) {
                    // bgn --------- offset_occur --- frame_disappear
                    fin = offsetOccur;
                } else {
                    // "bgn --- offset_occur --------- frame_disappear
                    fin = frameDisappear;
                }
                vector<float> tmp;
                tmp.push_back(bgn);
                tmp.push_back(fin);
                tmp.push_back(onsetShiftOutput[bgn]);
                tmp.push_back(offsetShiftOutput[fin]);
                tmp.push_back(velocityOutput[bgn]);
                outputTuples.push_back(tmp);
                bgn = 0;
                frameDisappear = 0;
                offsetOccur = 0;
            }
            if (bgn && (i - bgn >= 600 || i == onsetOutput.size() - 1)) {
                fin = i;
                vector<float> tmp;
                tmp.push_back(bgn);
                tmp.push_back(fin);
                tmp.push_back(onsetShiftOutput[bgn]);
                tmp.push_back(offsetShiftOutput[fin]);
                tmp.push_back(velocityOutput[bgn]);
                outputTuples.push_back(tmp);
                bgn = 0;
                frameDisappear = 0;
                offsetOccur = 0;
            }
        }
    }

    sort(outputTuples.begin(), outputTuples.end(), sortcol); 

    return outputTuples;
}

vector<vector<float>> RegressionPostProcessing::pedalDetectionWithOnsetOffsetRegress(
    vector<float> frameOutput, vector<float> offsetOutput,
    vector<float> offsetShiftOutput, float frameThreshold
) {
    /*
    * Process prediction array to pedal events information.
    
    *Args:
    *  frame_output: (frames_num,)
    *  offset_output: (frames_num,)
    *  offset_shift_output: (frames_num,)
    *  frame_threshold: float

    *Returns: 
    *  output_tuples: list of [bgn, fin, onset_shift, offset_shift], 
    *  e.g., [
    *    [1821, 1909, 0.4749851, 0.3048533], 
    *    [1909, 1947, 0.30730522, -0.45764327], 
    *    ...]
    */

    vector<vector<float>> outputTuples;
    int bgn;
    int frameDisappear;
    int offsetOccur;
    int fin;

    for (int i = 1; i < frameOutput.size(); i++) {
        if (frameOutput[i] >= frameThreshold && frameOutput[i] > frameOutput[i - 1]) {
            // Pedal onset detected
            if (!bgn) {
                bgn = i;
            }
        }

        if (bgn && i > bgn) {
            // If onset found, then search offset
            if (frameOutput[i] <= frameThreshold && !frameDisappear) {
                // Frame disappear detected
                frameDisappear = 1;
            }

            if (offsetOutput[i] == 1 && !offsetOccur) {
                // Offset detected
                offsetOccur = i;
            }

            if (offsetOccur) {
                fin = offsetOccur;
                vector<float> tmp;
                tmp.push_back(bgn);
                tmp.push_back(fin);
                tmp.push_back(0.0);
                tmp.push_back(offsetShiftOutput[bgn]);
                outputTuples.push_back(tmp);

                bgn = 0;
                frameDisappear = 0;
                offsetOccur = 0;
            }

            if (frameDisappear && i - frameDisappear >= 10) {
                // Offset note detected but frame disappear
                fin = frameDisappear;
                vector<float> tmp;
                tmp.push_back(bgn);
                tmp.push_back(fin);
                tmp.push_back(0.0);
                tmp.push_back(offsetShiftOutput[bgn]);
                outputTuples.push_back(tmp);

                bgn = 0;
                frameDisappear = 0;
                offsetOccur = 0;
            }
        }
    }

    sort(outputTuples.begin(), outputTuples.end(), sortcol);
    
    return outputTuples;
}

bool RegressionPostProcessing::isMonotonicNeighbour(vector<float> x, int n, int neighbour){
    /*
    * Detect if values are monotonic in both side of x[n].

    *    Args:
    *      x: (frames_num,)
    *      n: int
    *      neighbour: int

    *    Returns:
    *      monotonic: bool
    * */
    bool monotonic = true;
    for(int i = 0; i <= neighbour; i++){
            if(x[n-i] < x[n-i-1]) {
                monotonic = false;
            }
            if(x[n+i] < x[n+i+1]) {
                monotonic = false;
            }
    }
    return monotonic;
}

vector<vector<float>> RegressionPostProcessing::zerosLike(vector<vector<float>> inputArray) {
    vector<float> secondD(inputArray[0].size(), 0.0);
    vector<vector<float>> output(inputArray.size(), secondD);

    return output;
}

vector<float> RegressionPostProcessing::slicing(vector<vector<float>> inputVector, int index) {
    vector<float> out;

    for (vector<float> a : inputVector) {
        out.push_back(a[index]);
    }

    return out;
}

vector<float> RegressionPostProcessing::concatVectors(int num, int length) {
    vector<float> concated;

    for (int i = 0; i < length; i++)
    {
        concated.push_back((float)num);
    }

    return concated;
}

vector<float> RegressionPostProcessing::concatVectors(vector<float> arr, int length) {
    vector<float> output;

    for (int i = 0; i < length; i++)
    {
        for (float num : arr) {
        output.push_back(num);
        }
    }

    return output;
}

vector<float> RegressionPostProcessing::concatVectors(vector<float> arr1, vector<float> arr2) {
    for (float num : arr2) {
        arr1.push_back(num);
    }

    return arr1;
}

vector<vector<float>> RegressionPostProcessing::concatVectors(vector<vector<float>> arr1, vector<vector<float>> arr2) {
    for (vector<float> vec : arr2) {
        arr1.push_back(vec);
    }

    return arr1;
}

vector<float> RegressionPostProcessing::divideBy(vector<float> inputArr, int div) {
    for (int i = 0; i < inputArr.size(); i++) {
        inputArr[i] /= div;
    }

    return inputArr;
}

vector<vector<float>> RegressionPostProcessing::stack(vector<float> onsetTime, vector<float>offsetTime, vector<float> note, vector<float> velocity) {
    vector<vector<float>> output;

    for (int i = 0; i < note.size(); i++) {
        vector<float> tmp;
        tmp.push_back(onsetTime[i]);
        tmp.push_back(offsetTime[i]);
        tmp.push_back(note[i]);
        tmp.push_back(velocity[i]);
        output.push_back(tmp);
    }

    return output;
}

vector<vector<float>> RegressionPostProcessing::stack(vector<float> onsetTime, vector<float>offsetTime) {
    vector<vector<float>> output;

    for (int i = 0; i < onsetTime.size(); i++) {
        vector<float> tmp;
        tmp.push_back(onsetTime[i]);
        tmp.push_back(offsetTime[i]);
        output.push_back(tmp);
    }

    return output;
}

int main()
{
    ifstream file("2_sec_json.json");
    nlohmann::json jf = nlohmann::json::parse(file);
    unordered_map<string, vector<vector<float>>> jsonData = jf.get<unordered_map<string, vector<vector<float>>>>();
    cout << "asd\n";
    RegressionPostProcessing reg;
    unordered_map<string,vector<unordered_map<string, float>>> out = reg.outputMapToMidiEvents(jsonData);
    vector<unordered_map<string, float>> notes = out["est_note_events"];
    cout << notes.size() << endl;
    for (unordered_map<string, float> noteEvent : notes) {
        cout << "{\"onset_time\", " << noteEvent["onset_time"] << ",\t";
        cout << "\"offset_time\", " << noteEvent["offset_time"] << ",\t";
        cout << "\"midi_note\", " << noteEvent["midi_note"] << ",\t";
        cout << "\"velocity\", " << noteEvent["velocity"] << "},\n";
    }

    return 0;
}