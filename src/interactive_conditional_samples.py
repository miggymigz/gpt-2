#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model
import sample
import encoder


def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """
    # batch_size defaults to 1 if not specified
    if batch_size is None:
        batch_size = 1

    # the default nsamples is 1
    # asserts that user-specified nsamples should be divisible
    # by user-specified batch_size (the for-loop below reduces
    # the number of iterations by dividing nsamples by batch_size)
    assert nsamples % batch_size == 0

    # each GPT-2 have the same 'n_vocab'
    # so most likely, vocab.bpe and encoder.json will have the same values
    # maybe 'model_name' is passed here for directory organization
    # encoder is responsible for encoding words (also using Byte Pair Encoding)
    # In summary, sentences will be broken down into words.
    # Words may also be broken further down into byte pairs.
    # Then these byte pairs will be changed into the indices of the model's large vocabulary.
    enc = encoder.get_encoder(model_name)

    # this retrieves the GPT-2 variant's hyperparameters
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    # initialization of the length of each sample
    # should not exceed the `n_ctx` hyperparameter which is 1024
    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        # In tensorflow, model's inputs should use `tf.placeholder`.
        # This tells tensorflow to not compute gradients for it when minimizing its cost function.
        # Only variables defined using `tf.Variable`'s gradients will be computed.
        # Usually, these are the weights and biases that the model uses.
        # The shape [batch_size, None] allows the model to dynamically resize the placeholder.
        # The model's input changes for each step so `None` is appropriate.
        context = tf.placeholder(tf.int32, [batch_size, None])

        # Set seed to a constant to reproduce results
        np.random.seed(seed)
        tf.set_random_seed(seed)

        # Tensorflow graph initialization. All but `context` is ready and initialized.
        # Context will be set by calling sess.run() below and giving it `feeds` dict.
        # Note: This only initialize the graph and it does not actually execute the graph.
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        # The Saver class adds ops to save and restore variables to and from checkpoints.
        # It also provides convenience methods to run these ops.
        # Checkpoints are binary files in a proprietary format which map variable names to tensor values.
        # Restores weights and biases of the model variant to be used.
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        while True:
            # Asks for user input.
            # Allows users to write the first few sentences for the model to complete.
            raw_text = input("Model prompt >>> ")

            # Asks again for user input if user just pressed the enter key
            # without writing anything.
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")

            # Encodes the user-inputted sentence. The no. of tokens may not
            # match the actual number of encoded tokens (maybe because of BPE
            # or maybe its utf-8 encoding exceeds 7 bits).
            # The numbers in this list represents the indices of the word embeddings in the model's vocabulary.
            context_tokens = enc.encode(raw_text)

            # The total number of generated samples
            generated = 0

            # Sample generation is run by batch.
            # e.g., 4 samples and 2 batches will only have 2 iterations,
            # generating 2 samples for each iteration (batch).
            # If `nsamples` is more than 1, GPT-2 will generate more samples from the same input sentence.
            for _ in range(nsamples // batch_size):
                # Executes the graph created above giving context an actual value.
                # The context's axis=1's shape is determined by the length of the tokens.
                # The slicing part excludes the input tokens from the output token
                # so that when showing the sample, only the model's generated sample will be shown.
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]

                # Display the generated texts per sample
                for i in range(batch_size):
                    generated += 1

                    # Decode the output tokens which are just indices of the words in the model's vocabulary.
                    # Decoding also combines BPEs and re-encode UTF-8 sequences.
                    text = enc.decode(out[i])

                    print("=" * 40 + " SAMPLE " +
                          str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)


if __name__ == '__main__':
    fire.Fire(interact_model)
