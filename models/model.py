import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class Model(nn.Module):
    def __init__(
            self,
            device,
            num_topics,
            vocab_size,
            t_hidden_size,
            rho_size,
            emsize,
            theta_act,
            embeddings=None,
            train_embeddings=True,
            enc_drop=0.5,
            avg_words=0,
            debug_mode=False):
        super(Model, self).__init__()

        # define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)
        self.debug_mode = debug_mode
        self.avg_words = avg_words
        self.theta_act = self.get_activation(theta_act)

        self.device = device

        # define the word embedding matrix \rho
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            num_embeddings, emsize = embeddings.size()
            self.rho = embeddings.clone().float().to(self.device)

        # define the matrix containing the topic embeddings
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings)
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)

        # define variational distribution for \theta_{1:D} via amortizartion
        self.dropout = nn.Dropout(0.5)
        self.lstm_theta = nn.LSTM(input_size=rho_size, hidden_size=t_hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.q_theta = nn.Sequential(
            nn.Linear(t_hidden_size*2, t_hidden_size),
            self.theta_act,
            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
        )
        self.mu_q_theta = nn.Linear(t_hidden_size*2, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size*2, num_topics, bias=True)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            act = nn.Tanh()
            if self.debug_mode:
                print('Defaulting to tanh activation')
        return act

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows, real_bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        for counter, rows in enumerate(real_bows.cpu().numpy()):
            word_position = []
            for position, value in enumerate(rows):
                if value > 0:
                    word_2vec_value = self.rho[position].cpu().numpy().tolist()
                    word_position.append(word_2vec_value)
            word_position = np.array(word_position)
            if word_position.shape[0] < self.avg_words:
                word_position = np.pad(word_position, [(0, self.avg_words - word_position.shape[0]), (0, 0)], mode='constant', constant_values=0)
            else:
                word_position = word_position[:self.avg_words]
            if counter == 0:
                word_position_matrix = word_position
            else:    
                word_position_matrix = np.dstack([word_position_matrix, word_position])
        word_position_matrix = word_position_matrix.transpose(0, 2, 1)
        word_position_matrix = torch.from_numpy(word_position_matrix).to(self.device)

        emb = self.embedding_layer(bows.long().T)
        #output, dims = self.lstm_theta(emb.float())#bows.unsqueeze(0))
        output, dims = self.lstm_theta(word_position_matrix.float())
        output = self.dropout(output)
        q_theta = output
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * \
            torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        logit = self.alphas(self.rho)
        beta = F.softmax(
            logit, dim=0).transpose(
            1, 0)  # softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows, real_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows, real_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        if len(theta.shape) == 2:
            print(1)
        return theta, kld_theta

    def decode(self, theta, beta):
        res = torch.mm(torch.max(theta, dim=0)[0], beta)
        preds = torch.log(res + 1e-6)
        return preds

    def forward(self, bows, normalized_bows, theta=None, aggregate=True):
        # get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows, bows)
        else:
            kld_theta = None

        # get \beta
        beta = self.get_beta()
        
        # get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        recon_loss_pplxity = recon_loss
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta, recon_loss_pplxity
