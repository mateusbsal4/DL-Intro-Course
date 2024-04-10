from torch import nn
import torch
from ..util.transformer_util import positional_encoding, create_causal_mask, AttentionScoresSaver

# Module to save and visualize scores
SCORE_SAVER = AttentionScoresSaver()

class Transformer(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 eos_token_id: int,
                 hparams: dict = None,
                 max_length: int = 2048,
                 weight_tying: bool = True):
        """

        Args:
            vocab_size: Number of elements in the vocabulary
            eos_token_id: ID of the End-Of-Sequence Token - used in predict()
            weight_tying: Activate Weight Tying between Input Embedding and Output layer (default=True)
            max_length: Maximum sequence length (default=2048)

        Attributes:
            self.d_model: Dimension of Embedding (default=512)
            self.d_k: Dimension of Keys and Queries (default=64)
            self.d_v: Dimension of Values (default=64)
            self.n_heads: Number of Attention Heads (default=8)
            self.d_ff: Dimension of hidden layer (default=2048)
            self.n: Number of Encoder/Decoder Blocks (default=6)
            self.dropout: Dropout probability (default=0.1)
        """
        super().__init__()

        if hparams is None:
            hparams = {}
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.weight_tying = weight_tying

        self.d_model = hparams.get('d_model', 512)
        self.d_k = hparams.get('d_k', 64)
        self.d_v = hparams.get('d_v', self.d_k)
        self.n_heads = hparams.get('n_heads', 8)
        self.d_ff = hparams.get('d_ff', 2048)
        self.n = hparams.get('n', 6)
        self.dropout = hparams.get('dropout', 0.1)

        self.hparams = {
            'd_model': self.d_model,
            'd_k': self.d_k,
            'd_v': self.d_v,
            'd_ff': self.d_ff,
            'n_heads': self.n_heads,
            'n': self.n,
            'dropout': self.dropout
        }

        self.embedding = None
        self.encoder = None
        self.decoder = None
        self.output_layer = None

        ########################################################################
        # TODO:                                                                #
        #   Task 11: Initialize the the transformer!                           #
        #            You will need:                                            #
        #               - An embedding layer                                   #
        #               - An encoder                                           #
        #               - A decoder                                            #
        #               - An output layer                                      #
        #                                                                      #
        # Hint 11: Have a look at the output shape of the decoder and the      #
        #          output shape of the transformer model to figure out the     #
        #          dimensions of the output layer! We will not need a bias!    #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        if self.weight_tying:
            self.output_layer.weight = self.embedding.embedding.weight

    def forward(self,
                encoder_inputs: torch.Tensor,
                decoder_inputs: torch.Tensor,
                encoder_mask: torch.Tensor = None,
                decoder_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            encoder_inputs: Encoder Tokens Shape
            decoder_inputs: Decoder Tokens
            encoder_mask: Optional Padding Mask for Encoder Inputs
            decoder_mask: Optional Padding Mask for Decoder Inputs

        Returns:
                torch.Tensor: Logits of the Transformer Model
            
        Shape:
            - encoder_inputs: (batch_size, sequence_length_decoder)
            - decoder_inputs: (batch_size, sequence_length_encoder)
            - encoder_mask: (batch_size, sequence_length_encoder, sequence_length_encoder)
            - decoder_mask: (batch_size, sequence_length_decoder, sequence_length_decoder)
            - outputs: (batch_size, sequence_length_decoder, vocab_size)
        """

        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 11: Implement the forward pass of the transformer!            #
        #            You will need to:                                         #
        #               - Compute the encoder embeddings                       #
        #               - Compute the forward pass through the encoder         #
        #               - Compute the decoder embeddings                       #
        #               - Compute the forward pass through the decoder         #
        #               - Compute the output logits                            #
        #   Task 12: Pass on the encoder and decoder padding masks!            #
        #                                                                      #
        # Hints 12: Have a look at the forward pass of the encoder and decoder #
        #           to figure out which masks to pass on!                      #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs

    def predict(self,
                    encoder_input: torch.Tensor,
                    max_iteration_length: int = 100,
                    probabilistic: bool = False,
                    return_scores=False) -> tuple:
            """
            Predicts the output sequence given an input sequence using the Transformer model.

            Args:
                encoder_input (torch.Tensor): The input sequence to be encoded.
                max_iteration_length (int, optional): The maximum length of the output sequence. Defaults to 100.
                probabilistic (bool, optional): Whether to sample from the output distribution probabilistically. Defaults to False.
                return_scores (bool, optional): Whether to return the scores recorded during prediction. Defaults to False.

            Shape:
                - encoder_input: (sequence_length, d_model)

            Returns:
                tuple: A tuple containing the predicted output sequence and the recorded scores (if return_scores is True).
            """
            if return_scores:
                SCORE_SAVER.record_scores()

            # The Model only accepts batched inputs, so we have to add a batch dimension
            encoder_input = encoder_input.unsqueeze(0)

            self.eval()
            with torch.no_grad():

                # Compute the encoder embeddings
                encoder_input = self.embedding(encoder_input)

                # Run the embeddings through the encoder
                # We only have to do this once, since the input does not change!
                encoder_output = self.encoder(encoder_input)

                # Initialize the output sequence
                output_sequence = []

                for _ in range(max_iteration_length):

                    # Add the start token (or in our model it is the eos token) to the output sequence
                    # and add a batch dimension
                    decoder_input = torch.tensor([self.eos_token_id] + output_sequence).unsqueeze(0)

                    # Compute the decoder embeddings
                    decoder_input = self.embedding(decoder_input)

                    # Run the embeddings through the decoder
                    output = self.decoder(decoder_input, encoder_output)

                    # Compute the logits of the output layer
                    logits = self.output_layer(output).squeeze(0)

                    # We could run all logits through a softmax and would get the same result
                    # But we are going to just append the last output of the logits
                    # Remember - because of the causal masks, the predictions for the previous outputs never change!
                    last_logit = logits[-1]

                    # If probalistic is True, we sample from the output distribution and append the sample to the output sequence
                    if probabilistic:
                        output_distribution = torch.distributions.Categorical(logits=last_logit)
                        output = output_distribution.sample().item()
                        output_sequence.append(output)

                    # Else we just take the most likely output and append it to the output sequence
                    else:
                        output = torch.argmax(last_logit).item()
                        output_sequence.append(output)

                    # If we predicted the end of sequence token, we stop
                    if output_sequence[-1] is self.eos_token_id:
                        break

            return output_sequence, SCORE_SAVER.get_scores()


class Embedding(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 max_length: int,
                 dropout: float = 0.0):
        """

        Args:
            vocab_size: Number of elements in the vocabulary
            d_model: Dimension of Embedding
            max_length: Maximum sequence length
        """
        super().__init__()

        self.embedding = None
        self.pos_encoding = None
        self.dropout = None

        ########################################################################
        # TODO:                                                                #
        #   Task 1: Initialize the embedding layer (torch.nn implementation)   #
        #   Task 4: Initialize the positional encoding layer.                  #
        #   Task 13: Initialize the dropout layer (torch.nn implementation)    #
        #                                                                      #
        # Hints 1:                                                             #
        #       - Have a look at pytorch embedding module                      #
        # Hints 4:                                                             #
        #       - We have implemented the positional encoding in               #
        #         exercise_code/util/transformer_util.py for you.              #
        #       - You can use it simply by calling positional_encoding(...)    #
        #       - Initialize it using d_model and max_length                   #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        # We will convert it into a torch parameter module for you! You can treat it like a normal tensor though!
        if self.pos_encoding is not None:
            self.pos_encoding = nn.Parameter(data=self.pos_encoding, requires_grad=False)

    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        """
        The forward function takes in tensors of token ids and transforms them into vector embeddings. 
        It then adds the positional encoding to the embeddings, and if configured, performs dropout on the layer!

        Args:
            inputs: Batched Sequence of Token Ids

        Shape:
            - inputs: (batch_size, sequence_length)
            - outputs: (batch_size, sequence_length, d_model)
        """

        outputs = None

        # Use fancy indexing to extract the positional encodings until position sequence_length
        sequence_length = inputs.shape[-1]
        pos_encoding = 0
        if self.pos_encoding is not None:
            pos_encoding = self.pos_encoding[:sequence_length]

        ########################################################################
        # TODO:                                                                #
        #   Task 1: Compute the outputs of the embedding layer                 #
        #   Task 4: Add the positional encoding to the output                  #
        #   Task 13: Add dropout as a final step                               #
        #                                                                      #
        # Hint 4: We have already extracted them for you, all you have to do   #
        #         is add them to the embeddings!                               #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs


class ScaledDotAttention(nn.Module):

    def __init__(self,
                 d_k,
                 dropout: float = 0.0):
        """

        Args:
            d_k: Dimension of Keys and Queries
            dropout: Dropout probability
        """
        super().__init__()
        self.d_k = d_k

        self.softmax = None
        self.dropout = None

        ########################################################################
        # TODO:                                                                #
        #   Task 2: Initialize the softmax layer (torch.nn implementation)     #
        #   Task 13: Initialize the dropout layer (torch.nn implementation)    #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the scaled dot attention given query, key and value inputs. Stores the scores in SCORE_SAVER for
        visualization

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs
            mask: Optional Causal or Padding Boolean Mask

        Shape:
            - q: (*, sequence_length_queries, d_model)
            - k: (*, sequence_length_keys, d_model)
            - v: (*, sequence_length_keys, d_model)
            - mask: (*, sequence_length_queries, sequence_length_keys)
            - outputs: (*, sequence_length_queries, d_v)
        """
        scores = None
        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 2:                                                            #
        #       - Calculate the scores using the queries and keys              #
        #       - Normalise the scores using the softmax function              #
        #       - Compute the updated embeddings and return the output         #
        #   Task 8:                                                            #
        #       - Add a negative infinity mask if a mask is given              #
        #   Task 13:                                                           #
        #       - Add dropout to the scores right BEFORE the final outputs     #
        #         (scores * V) are calculated                                  #
        #                                                                      #
        # Hint 2:                                                              #
        #       - torch.transpose(x, dim_1, dim_2) swaps the dimensions dim_1  #
        #         and dim_2 of the tensor x!                                   #
        #       - Later we will insert more dimensions into *, so how could    #
        #         index these dimensions to always get the right ones?         #
        #       - Also dont forget to scale the scores as discussed!           #
        # Hint 8:                                                              #
        #       - Have a look at Tensor.masked_fill_() or use torch.where()    #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        SCORE_SAVER.save(scores)

        return outputs


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            dropout: Dropout probability
        """
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.weights_q = None
        self.weights_k = None
        self.weights_v = None
        self.attention = None
        self.project = None
        self.dropout = None

        ########################################################################
        # TODO:                                                                #
        #   Task 3:                                                            #
        #       -Initialize all weight layers as linear layers                 #
        #       -Initialize the ScaledDotAttention                             #
        #       -Initialize the projection layer as a linear layer             #
        #  Task 13:                                                            #
        #       -Initialize the dropout layer (torch.nn implementation)        #
        #                                                                      #
        # Hints 3:                                                             #
        #       - Instead of initializing several weight layers for each head, #
        #         you can create one large weight matrix. This speed up        #
        #         the forward pass, since we dont have to loop through all     #
        #         heads!                                                       #
        #       - All linear layers should only be a weight without a bias!    #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs
            mask: Optional Causal or Padding Mask

        Shape:
            - q: (batch_size, sequence_length_queries, d_model)
            - k: (batch_size, sequence_length_keys, d_model)
            - v: (batch_size, sequence_length_keys, d_model)
            - v: (batch_size, sequence_length_queries, sequence_length_keys)
            - outputs: (batch_size, sequence_length_queries, d_model)
        """

        # You will need these here!
        batch_size, sequence_length_queries, _ = q.size()
        _, sequence_length_keys, _ = k.size()

        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 3:                                                            #
        #       - Pass q,k and v through the linear layer                      #
        #       - Split the last dimensions into n_heads and d_k od d_v        #
        #       - Swap the dimensions so that the shape matches the required   #
        #         input shapes of the ScaledDotAttention layer                 #
        #       - Pass them through the ScaledDotAttention layer               #
        #       - Swap the dimensions of the output back                       #
        #       - Combine the last two dimensions again                        #
        #       - Pass the outputs through the projection layer                #
        #   Task 8:                                                            #
        #       - If a mask is given, add an empty dimension at dim=1          #
        #       - Pass the mask to the ScaledDotAttention layer                #
        #  Task 13:                                                            #
        #       - Add dropout as a final step after the projection layer       #
        #                                                                      #
        # Hints 3:                                                             #
        #       - It helps to write down which dimensions you want to have on  #
        #         paper!                                                       #
        #       - Above the todo, we have already extracted the batch_size and #
        #         the sequence lengths for you!                                #
        #       - Use reshape() to split or combine dimensions                 #
        #       - Use transpose() again to swap dimensions                     #
        # Hints 8:                                                             #
        #       - Use unsqueeze() to add dimensions at the correct location    #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs


class FeedForwardNeuralNetwork(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_ff: Dimension of hidden layer
            dropout: Dropout probability
        """
        super().__init__()

        self.linear_1 = None
        self.relu = None
        self.linear_2 = None
        self.dropout = None

        ########################################################################
        # TODO:                                                                #
        #   Task 5: Initialize the feed forward network                        #
        #   Task 13: Initialize the dropout layer (torch.nn implementation)    #
        #                                                                      #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        """

        Args:
            inputs: Inputs to the Feed Forward Network

        Shape:
            - inputs: (batch_size, sequence_length_queries, d_model)
            - outputs: (batch_size, sequence_length_queries, d_model)
        """
        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 5: Implement forward pass of feed forward layer               #
        #   Task 13: Pass the output through a dropout layer as a final step   #
        #                                                                      #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs


class EncoderBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            d_ff: Dimension of hidden layer
            dropout: Dropout probability
        """
        super().__init__()

        self.multi_head = None
        self.layer_norm1 = None
        self.ffn = None
        self.layer_norm2 = None

        ########################################################################
        # TODO:                                                                #
        #   Task 6: Initialize an Encoder Block                                #
        #           You will need:                                             #
        #                           - Multi-Head Self-Attention layer          #
        #                           - Layer Normalization                      #
        #                           - Feed forward neural network layer        #
        #                           - Layer Normalization                      #
        #                                                                      #
        # Hint 6: Check out the pytorch layer norm module                      #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                inputs: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            inputs: Inputs to the Encoder Block
            pad_mask: Optional Padding Mask

        Shape:
            - inputs: (batch_size, sequence_length, d_model)
            - pad_mask: (batch_size, sequence_length, sequence_length)
            - outputs: (batch_size, sequence_length, d_model)
        """
        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 6: Implement the forward pass of the encoder block            #
        #   Task 12: Pass on the padding mask                                  #
        #                                                                      #
        # Hint 6: Don't forget the residual connection! You can forget about   #
        #         the pad_mask for now!                                        #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs


class Encoder(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 d_ff: int,
                 n: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            d_ff: Dimension of hidden layer
            n: Number of Encoder Blocks
            dropout: Dropout probability
        """
        super().__init__()

        self.stack = nn.ModuleList([EncoderBlock(d_model=d_model,
                                                 d_k=d_k,
                                                 d_v=d_v,
                                                 n_heads=n_heads,
                                                 d_ff=d_ff,
                                                 dropout=dropout) for _ in range(n)])


    def forward(self,
                inputs: torch.Tensor,
                encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            inputs: Inputs to the Encoder Stack
            encoder_mask: Optional Padding Mask for Encoder Inputs

        Shape:
            - inputs: (batch_size, sequence_length, d_model)
            - encoder_mask: (batch_size, 1, sequence_length)
            - outputs: (batch_size, sequence_length, d_model)
        """

        # This is just so we can loop through the encoder blocks nicer - it is complettly unnecessary!
        outputs = inputs

        # Loop through the encoder blocks
        for encoder in self.stack:
            outputs = encoder(outputs, encoder_mask)

        return outputs


class DecoderBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            d_ff: Dimension of hidden layer
            dropout: Dropout probability
        """
        super().__init__()

        self.causal_multi_head = None
        self.layer_norm1 = None
        self.cross_multi_head = None
        self.layer_norm2 = None
        self.ffn = None
        self.layer_norm3 = None

        ########################################################################
        # TODO:                                                                #
        #   Task 9: Initialize an Decoder Block                                #
        #            You will need:                                            #
        #                           - Causal Multi-Head Self-Attention layer   #
        #                           - Layer Normalization                      #
        #                           - Multi-Head Cross-Attention layer         #
        #                           - Layer Normalization                      #
        #                           - Feed forward neural network layer        #
        #                           - Layer Normalization                      #
        #                                                                      #
        # Hint 9: Check out the pytorch layer norm module                      #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                inputs: torch.Tensor,
                context: torch.Tensor,
                causal_mask: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            inputs: Inputs from the Decoder
            context: Context from the Encoder
            causal_mask: Mask used for Causal Self Attention
            pad_mask: Optional Padding Mask used for Cross Attention

        Shape: 
            - inputs: (batch_size, sequence_length_decoder, d_model)
            - context: (batch_size, sequence_length_encoder, d_model)
            - causal_mask: (batch_size, sequence_length_decoder, sequence_length_decoder)
            - pad_mask: (batch_size, sequence_length_decoder, sequence_length_encoder)
            - outputs: (batch_size, sequence_length_decoder, d_model)
        """
        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 9: Implement the forward pass of the decoder block            #
        #   Task 12: Pass on the padding mask                                  #
        #                                                                      #
        # Hint 9:                                                              #
        #       - Don't forget the residual connections!                       #
        #       - Remember where we need the causal mask, forget about the     #
        #         other mask for now!                                          #
        # Hints 12:                                                            #
        #       - We have already combined the causal_mask with the pad_mask   #
        #         for you, all you have to do is pass it on to the "other"     #
        #         module                                                       #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs


class Decoder(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 d_ff: int,
                 n: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            d_ff: Dimension of hidden layer
            n: Number of Decoder Blocks
            dropout: Dropout probability
        """
        super().__init__()

        self.stack = nn.ModuleList([DecoderBlock(d_model=d_model,
                                                 d_k=d_k,
                                                 d_v=d_v,
                                                 n_heads=n_heads,
                                                 d_ff=d_ff,
                                                 dropout=dropout) for _ in range(n)])
        

    def forward(self,
                inputs: torch.Tensor,
                context: torch.Tensor,
                decoder_mask: torch.Tensor = None,
                encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            inputs: Inputs from the Decoder
            context: Context from the Encoder
            decoder_mask: Optional Padding Mask for Decoder Inputs
            encoder_mask: Optional Padding Mask for Encoder Inputs

        Shape: 
            - inputs: (batch_size, sequence_length_decoder, d_model)
            - context: (batch_size, sequence_length_encoder, d_model)
            - decoder_mask: (batch_size, sequence_length_decoder, sequence_length_decoder)
            - encoder_mask: (batch_size, sequence_length_encoder, sequence_length_encoder)
            - outputs: (batch_size, sequence_length_decoder, d_model)
        """

        # Create a causal mask for the decoder
        causal_mask = create_causal_mask(inputs.shape[-2]).to(inputs.device)

        # Combine the causal mask with the decoder mask - We haven't discussed this yet so don't worry about it!
        if decoder_mask is not None:
            causal_mask = causal_mask * decoder_mask

        # This is just so we can loop through the decoder blocks nicer - it is complettly unnecessary!
        outputs = inputs

        # Loop through the decoder blocks
        for decoder in self.stack:
            outputs = decoder(outputs, context, causal_mask, encoder_mask)

        return outputs
