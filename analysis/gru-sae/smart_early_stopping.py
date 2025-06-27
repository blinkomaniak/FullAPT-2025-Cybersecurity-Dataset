"""
Smart Early Stopping Callback for LSTM-SAE Training

This callback implements intelligent early stopping that requires SIGNIFICANT improvements
to continue training, preventing endless training on millicimal improvements.
"""

import tensorflow as tf
import numpy as np
from collections import deque

class SmartEarlyStopping(tf.keras.callbacks.Callback):
    """
    Advanced early stopping that requires meaningful improvements to continue training.
    
    Stops training if:
    1. No significant improvement for 'patience' epochs
    2. Improvement rate over 'window' epochs is too slow
    3. Total improvement over 'window' epochs is below threshold
    """
    
    def __init__(self, 
                 monitor='val_loss',
                 min_improvement_pct=0.5,      # Minimum 0.5% relative improvement required
                 min_improvement_abs=0.0005,   # Minimum absolute improvement required  
                 patience=5,                   # Epochs to wait for significant improvement
                 improvement_window=10,        # Window to evaluate improvement rate
                 min_window_improvement=2.0,   # Minimum % improvement over window
                 min_improvement_rate=0.1,     # Minimum % improvement per epoch over window
                 baseline_epochs=5,            # Epochs to establish baseline
                 restore_best_weights=True,
                 verbose=1):
        
        super().__init__()
        self.monitor = monitor
        self.min_improvement_pct = min_improvement_pct / 100.0  # Convert to decimal
        self.min_improvement_abs = min_improvement_abs
        self.patience = patience
        self.improvement_window = improvement_window
        self.min_window_improvement = min_window_improvement / 100.0
        self.min_improvement_rate = min_improvement_rate / 100.0
        self.baseline_epochs = baseline_epochs
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # State tracking
        self.best_loss = np.inf
        self.best_epoch = 0
        self.wait = 0
        self.loss_history = deque(maxlen=improvement_window)
        self.baseline_loss = None
        self.stopped_epoch = 0
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        self.best_loss = np.inf
        self.best_epoch = 0
        self.wait = 0
        self.loss_history.clear()
        self.baseline_loss = None
        self.stopped_epoch = 0
        self.best_weights = None
        
        if self.verbose:
            print(f"\n🧠 Smart Early Stopping Configuration:")
            print(f"   Minimum improvement: {self.min_improvement_pct*100:.1f}% or {self.min_improvement_abs:.6f}")
            print(f"   Patience: {self.patience} epochs")
            print(f"   Window analysis: {self.improvement_window} epochs")
            print(f"   Required window improvement: {self.min_window_improvement*100:.1f}%")
            print(f"   Required improvement rate: {self.min_improvement_rate*100:.1f}%/epoch")
    
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            return
            
        self.loss_history.append(current_loss)
        
        # Establish baseline in first few epochs
        if epoch < self.baseline_epochs:
            if self.baseline_loss is None or current_loss < self.baseline_loss:
                self.baseline_loss = current_loss
            return
        
        # Check for significant improvement
        is_significant_improvement = self._is_significant_improvement(current_loss)
        
        if is_significant_improvement:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            
            if self.verbose:
                improvement_pct = ((self.loss_history[-2] - current_loss) / self.loss_history[-2]) * 100
                print(f"✅ Significant improvement: {improvement_pct:.2f}% (epoch {epoch+1})")
        else:
            self.wait += 1
            if self.verbose and len(self.loss_history) >= 2:
                change_pct = ((self.loss_history[-2] - current_loss) / self.loss_history[-2]) * 100
                print(f"⚠️ Minor improvement: {change_pct:.3f}% - patience {self.wait}/{self.patience}")
        
        # Check stopping conditions
        should_stop, reason = self._should_stop(epoch)
        
        if should_stop:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            
            if self.verbose:
                print(f"\n🛑 Smart Early Stopping triggered at epoch {epoch+1}")
                print(f"   Reason: {reason}")
                print(f"   Best loss: {self.best_loss:.6f} (epoch {self.best_epoch+1})")
                
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)
                if self.verbose:
                    print(f"   Restored weights from epoch {self.best_epoch+1}")
    
    def _is_significant_improvement(self, current_loss):
        """Check if current loss represents significant improvement"""
        if len(self.loss_history) < 2:
            return current_loss < self.best_loss
            
        previous_loss = self.loss_history[-2]
        
        # Absolute improvement check
        abs_improvement = previous_loss - current_loss
        if abs_improvement < self.min_improvement_abs:
            return False
            
        # Relative improvement check  
        rel_improvement = abs_improvement / previous_loss
        if rel_improvement < self.min_improvement_pct:
            return False
            
        # Must also be better than best seen so far
        return current_loss < self.best_loss
    
    def _should_stop(self, epoch):
        """Determine if training should stop and provide reason"""
        
        # Standard patience check
        if self.wait >= self.patience:
            return True, f"No significant improvement for {self.patience} epochs"
        
        # Window-based analysis (only after sufficient history)
        if len(self.loss_history) >= self.improvement_window:
            window_start_loss = self.loss_history[0]
            current_loss = self.loss_history[-1]
            
            # Check total improvement over window
            window_improvement = (window_start_loss - current_loss) / window_start_loss
            if window_improvement < self.min_window_improvement:
                return True, f"Improvement over {self.improvement_window} epochs ({window_improvement*100:.2f}%) below threshold ({self.min_window_improvement*100:.1f}%)"
            
            # Check improvement rate over window
            epochs_in_window = len(self.loss_history)
            improvement_rate = window_improvement / epochs_in_window
            if improvement_rate < self.min_improvement_rate:
                return True, f"Improvement rate ({improvement_rate*100:.3f}%/epoch) below threshold ({self.min_improvement_rate*100:.1f}%/epoch)"
        
        return False, None
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose:
            print(f"\n📊 Training stopped early at epoch {self.stopped_epoch + 1}")
            print(f"   Best validation loss: {self.best_loss:.6f}")
            if self.baseline_loss:
                total_improvement = ((self.baseline_loss - self.best_loss) / self.baseline_loss) * 100
                print(f"   Total improvement: {total_improvement:.2f}% from baseline")


class AdaptiveEarlyStopping(tf.keras.callbacks.Callback):
    """
    Simpler adaptive early stopping that adjusts thresholds based on loss magnitude
    """
    
    def __init__(self,
                 monitor='val_loss', 
                 base_patience=5,
                 min_improvement_ratio=0.001,  # 0.1% minimum improvement
                 adaptive_threshold=True,
                 restore_best_weights=True,
                 verbose=1):
        
        super().__init__()
        self.monitor = monitor
        self.base_patience = base_patience
        self.min_improvement_ratio = min_improvement_ratio
        self.adaptive_threshold = adaptive_threshold
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = np.inf
        self.best_epoch = 0
        self.wait = 0
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        self.best_loss = np.inf
        self.best_epoch = 0
        self.wait = 0
        self.best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            return
        
        # Calculate adaptive threshold based on current loss magnitude
        if self.adaptive_threshold:
            # For losses around 0.01, require 0.0001 improvement (1%)
            # For losses around 0.001, require 0.00001 improvement (1%)
            min_improvement = max(current_loss * self.min_improvement_ratio, 1e-6)
        else:
            min_improvement = self.min_improvement_ratio
        
        # Check for meaningful improvement
        improvement = self.best_loss - current_loss
        
        if improvement > min_improvement:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
                
            if self.verbose:
                improvement_pct = (improvement / self.best_loss) * 100
                print(f"✅ Meaningful improvement: {improvement:.6f} ({improvement_pct:.2f}%)")
        else:
            self.wait += 1
            if self.verbose:
                print(f"⚠️ Insufficient improvement: {improvement:.6f} < {min_improvement:.6f} - patience {self.wait}/{self.base_patience}")
                
        if self.wait >= self.base_patience:
            self.model.stop_training = True
            if self.verbose:
                print(f"\n🛑 Adaptive Early Stopping at epoch {epoch+1}")
                print(f"   Best loss: {self.best_loss:.6f} (epoch {self.best_epoch+1})")
                
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)


# Usage example and testing functions
def create_smart_callbacks(strategy='smart', **kwargs):
    """Factory function to create appropriate early stopping callbacks"""
    
    if strategy == 'smart':
        return [
            SmartEarlyStopping(**kwargs),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1
            )
        ]
    
    elif strategy == 'adaptive':
        return [
            AdaptiveEarlyStopping(**kwargs),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
            )
        ]
    
    elif strategy == 'standard':
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=kwargs.get('patience', 5),
                restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
            )
        ]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    # Test the callback with synthetic data
    print("Smart Early Stopping Callback - Ready for use!")
    print("\nUsage:")
    print("from smart_early_stopping import SmartEarlyStopping, create_smart_callbacks")
    print("\n# Option 1: Direct usage")
    print("callback = SmartEarlyStopping(min_improvement_pct=0.5, patience=5)")
    print("\n# Option 2: Factory function")  
    print("callbacks = create_smart_callbacks('smart', min_improvement_pct=1.0, patience=7)")