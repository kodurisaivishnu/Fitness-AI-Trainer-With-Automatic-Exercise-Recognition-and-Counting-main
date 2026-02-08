# Research References

## Primary reference (base paper)

**Real-Time Fitness Exercise Classification and Counting from Video Frames**  
Riccardo Riccio  
arXiv:2411.11548 [cs.CV] (2024)

- **Link:** https://arxiv.org/abs/2411.11548
- **PDF:** https://arxiv.org/pdf/2411.11548
- **Abstract:** Proposes a BiLSTM-based model using invariant features (joint angles + normalized distances) and 30-frame sequences for real-time exercise classification. Achieves >99% test accuracy on squat, push-up, shoulder press, and bicep curl.
- **Dataset:** Kaggle Real-Time Exercise Recognition Dataset, InfiniteRep (synthetic), and other sources.
- **Code/Data:** https://github.com/RiccardoRiccio/Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting

### BibTeX (for your paper)

```bibtex
@article{riccio2024realtime,
  title={Real-Time Fitness Exercise Classification and Counting from Video Frames},
  author={Riccio, Riccardo},
  journal={arXiv preprint arXiv:2411.11548},
  year={2024},
  url={https://arxiv.org/abs/2411.11548},
  note={cs.CV}
}
```

---

## Related work (for literature review and citations)

1. **MediaPipe Pose**
   - Google; used for body landmark extraction.
   - https://google.github.io/mediapipe/solutions/pose

2. **InfiniteRep dataset**
   - Synthetic exercise videos (avatars).
   - Often cited with exercise recognition and rep counting.

3. **Kaggle – Real-Time Exercise Recognition Dataset**
   - Riccardo Riccio (and others).
   - https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset

4. **LSTM/BiLSTM for action recognition**
   - Standard references for using RNNs on skeleton/pose sequences for action or exercise recognition.

5. **Exercise repetition counting**
   - Search terms: "repetition counting", "exercise counting", "pose estimation fitness" (e.g. on Google Scholar / IEEE Xplore / ACM DL).

---

## How to cite in your paper

- **Methodology / system design:** Cite the primary reference (Riccio, arXiv:2411.11548) for the BiLSTM design, 30-frame sequences, and invariant features (angles + normalized distances).
- **Dataset:** Cite the Kaggle dataset and, if you use synthetic data, InfiniteRep.
- **Pose pipeline:** Cite MediaPipe for pose estimation.

Use the BibTeX above in your references section and refer to it in the text (e.g. “Following the approach of Riccio [1], we use a BiLSTM on 30-frame sequences of joint angles and normalized distances.”).
