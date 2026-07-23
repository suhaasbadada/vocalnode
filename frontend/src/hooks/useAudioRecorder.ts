import { useState, useRef, useCallback } from 'react';

interface UseAudioRecorderProps {
  onComplete?: (file: File) => void;
}

export const useAudioRecorder = ({ onComplete }: UseAudioRecorderProps = {}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const timerInterval = useRef<number | null>(null);

  const stopRecording = useCallback(() => {
    return new Promise<File | null>((resolve) => {
      if (!mediaRecorder.current || mediaRecorder.current.state === 'inactive') {
        resolve(null);
        return;
      }

      mediaRecorder.current.onstop = () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' });
        const file = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });
        
        // Cleanup
        if (timerInterval.current) clearInterval(timerInterval.current);
        mediaRecorder.current?.stream.getTracks().forEach(track => track.stop());
        setIsRecording(false);
        setRecordingTime(0);
        
        if (onComplete) {
          onComplete(file);
        }
        resolve(file);
      };

      mediaRecorder.current.stop();
    });
  }, [onComplete]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };

      mediaRecorder.current.start();
      setIsRecording(true);
      setRecordingTime(0);

      timerInterval.current = window.setInterval(() => {
        setRecordingTime((prev) => {
          if (prev >= 9) {
            // Trigger internal stop, which resolves the file and calls onComplete
            stopRecording();
            return 10;
          }
          return prev + 1;
        });
      }, 1000);

    } catch (err) {
      console.error('Error accessing microphone:', err);
      alert('Could not access microphone. Please ensure you have granted permission.');
    }
  }, [stopRecording]);

  return { isRecording, recordingTime, startRecording, stopRecording };
};
