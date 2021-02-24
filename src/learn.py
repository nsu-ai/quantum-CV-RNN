import tensorflow as tf
from statistics import mean
import matplotlib.pyplot as plt
from .quantum_basic_blocks import trace


# metric must be mean per batch
def learn(model, train_ds, test_ds, loss_fn, opt, epochs, metric_fn = None, record_steps = False, record_trace = False, skip = 1):
    train_loss_record = list()
    test_loss_record = list()
    metric_record = list()
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        # do opt
        for step, (x_batch, y_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                out = model(x_batch)
                loss = loss_fn(y_batch, out)

            if record_steps and step % skip == 0:
                if record_trace:
                    state = model(x_batch, return_state = True)
                    print(
                        'Training loss and mean trace (for one batch) at step {}: {:.4f}, {:.4f}'.format(
                            step, float(loss), trace(state))
                    )
                else:
                    print(
                    'Training loss (for one batch) at step {}: {:.4f},'.format(
                        step, float(loss))
                    )
                
                print('Seen so far: {} samples'.format((step + 1) * y_batch.shape[0]))
            
            gradients = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables))
            
        losses_tr = list()
        traces_tr = list()
        metric_te = list()
        for step, (x_batch, y_batch) in enumerate(train_ds):
            out = model(x_batch)
            loss = loss_fn(y_batch, out)
            losses_tr.append(loss.numpy())
            if record_trace:
                state = model(x_batch, return_state = True)
                traces_tr += trace.numpy().tolist()
        
        losses_te = list()
        traces_te = list()
        for step, (x_batch, y_batch) in enumerate(test_ds):
            out = model(x_batch)
            loss = loss_fn(y_batch, out)
            losses_te.append(loss.numpy())
            if record_trace:
                state = model(x_batch, return_state = True)
                traces_te += trace.numpy().tolist()
            if metric_fn:
                metric_te.append(metric_fn(out, y_batch))
        
        losses_tr = mean(losses_tr)
        losses_te = mean(losses_te)
        print('Train and test losses: {:.4f}, {:.4f}'.format(losses_tr, losses_te))
        if record_trace:
            print('Train and test mean traces: {:.4f}, {:.4f}'.format(mean(traces_tr), mean(traces_te)))
            print('Train and test min traces: {:.4f}, {:.4f}'.format(min(traces_tr), min(traces_te)))
        
        train_loss_record.append(losses_tr)
        test_loss_record.append(losses_te)
        if metric_fn:
            metric_te = mean(metric_te)
            print('Metric: {:.4f}'.format(metric_te))
            metric_record.append(metric_te)
        
    return tf.stack(train_loss_record), tf.stack(test_loss_record), tf.stack(metric_record)


def plot(te_r, metric):
    if metric.shape[0]:
        fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
        fig.suptitle('Training Metrics')
        axes[0].set_ylabel("Loss", fontsize=14)
        axes[0].plot(te_r)
        
        # axes[1].set_ylim([0, 1.05])
        axes[1].set_ylabel("Metric", fontsize=14)
        axes[1].set_xlabel("Epoch", fontsize=14)
        axes[1].plot(metric)
    else:
        fig, axes = plt.subplots(1, sharex=True, figsize=(12, 4))
        fig.suptitle('Training Metric')
        axes.set_ylabel("Loss", fontsize=14)
        axes.plot(te_r)
        
    plt.show()
