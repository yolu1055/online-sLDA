import sys, re, time, string
import numpy as n
from scipy.special import gammaln, psi
from scipy import linalg as LA
import parse_document

n.random.seed(100000001)
meanchangethresh = 0.001

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

class onlineSLDA_test:
    def __init__(self, V, K, C, D, mu, _lambda, max_it, alpha):
        self.iterations = 0
        self._K = K
        self._V = V
        self._C = C
        self._D = D
        self._max_it = max_it
        self._alpha = alpha
        #self._eta = eta # dirichlet parameters
        self._lambda = _lambda
        self._mu = mu
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._likelihoods = list()
        self._scores = n.zeros((D, self._C))
        self._predictions = list()
        
    def do_e_step(self, wordids, wordcts ):
        likelihood = 0.0
        #batchD = len(wordids)
        gamma = 1*n.random.gamma(100., 1./100., (self._D, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)
        for d in range(0, self._D):
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d,:]
            Elogthetad = Elogtheta[d,:]
            expElogthetad = expElogtheta[d,:]
            expElogbetad = self._expElogbeta[:,ids]
            Elogbetad = self._Elogbeta[:,ids]
            phi = n.ones((len(ids), self._K))/float(self._K)
            
            for it in range(0, self._max_it):
                lastgamma = gammad
                gammad = self._alpha +\
                 n.sum (phi.T * cts, axis = 1)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                
                phi = (expElogthetad * expElogbetad.T)
                phinorm = n.sum(phi, axis = 1) + 1e-100
                phi = phi / phinorm[:,n.newaxis]
                #phi = (phi.T / phinorm).T
                #phi_old = phi
                #nphi = (phi.T * cts).T
                
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            
            gamma[d, :] = gammad
            likelihood = likelihood + self.cal_loacllieklihood(phi, cts, Elogthetad, Elogbetad)
            self._scores[d,:] = n.dot(self._mu, n.average(phi.T * cts, axis = 1))
            self._predictions.append(n.argmax(self._scores[d,:]))
        
    
    def cal_loacllieklihood(self, phi, cts, Elogthetad, Elogbetad):
        likelihood = 0.0
        nphi = (phi.T * cts).T
        Elogpz_qz = n.sum(n.sum(nphi * (Elogthetad - n.log(phi))))
        Elogpw = n.sum(n.sum(nphi * Elogbetad.T))
        likelihood = Elogpz_qz + Elogpw
            
        return likelihood
    
    def saveresults(self, it):
        n.save("scores.txt", self._scores)
        f = open("./predictions_%d.txt"%it, "w")
        for d in range(0, self._D):
            f.write(str(self._predictions[d]))
            f.write("\n") 
        f.close()
        
    def accuracy(self, goldlabel):
        right = 0
        for d in range(0, self._D):
            if (self._predictions[d] == goldlabel[d]):
                right = right + 1
        accuracy = float(right) / float(self._D)
        return accuracy 
                    


class onlineSLDA_train:
    def __init__(self,V, K, D, C, mu, max_it, alpha, eta, tau1, tau2, kappa):
        self._iterations = 0
        self._D = D
        self._K = K
        self._V = V
        self._C = C # number of categories
        #self._mu = n.zeros((self._C, self._K))
        self._mu = 1*n.random.gamma(100., 1./100., (self._C, self._K)) # softmax parameters
        self._max_it = max_it
        self._alpha = alpha
        self._eta = eta # dirichlet parameters
        self._tau1 = tau1
        self._tau2 = tau2
        self._kappa = kappa
        self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._V))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._likelihoods = list()
        
        
    def do_e_step(self, wordids, wordcts, labels):
        likelihood = 0.0
        sstats = n.zeros(self._lambda.shape)
        grad_mu = n.zeros(self._mu.shape)
        batchD = len(wordids)
        gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)
        #phiD = list()
        for d in range(0, batchD):
            ids = wordids[d]
            cts = wordcts[d]
            label = labels[d]
            N = float(sum(cts))
            gammad = gamma[d,:]
            Elogthetad = Elogtheta[d,:]
            expElogthetad = expElogtheta[d,:]
            expElogbetad = self._expElogbeta[:,ids]
            Elogbetad = self._Elogbeta[:,ids]
            expmu = n.exp((1.0/N)*self._mu)
            expmud = expmu[label, :]
            phi = n.ones((len(ids), self._K))/float(self._K)
            (h_phiprod, h) = self.calculatesfaux(phi, expmu, cts)
            
            for it in range(0, self._max_it):
                lastgamma = gammad
                gammad = self._alpha +\
                 n.sum (phi.T * cts, axis = 1)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                
                phi = (expElogthetad * expElogbetad.T) * expmud / n.exp(h/h_phiprod)
                phinorm = n.sum(phi, axis = 1) + 1e-100
                phi = (phi / phinorm[:,n.newaxis]) +1e-100
                #phi = (phi.T / phinorm).T
                #phi_old = phi
                #nphi = (phi.T * cts).T
                (h_phiprod, h) = self.calculatesfaux(phi, expmu, cts)
                
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            
            gamma[d, :] = gammad
            sstats[:,ids] += phi.T * cts
            
            #sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)
            grad_mu = grad_mu + self.calgradmu(phi, expmu, cts, label)
            likelihood = likelihood +\
             self.cal_locallikelihood(phi, cts, Elogthetad, Elogbetad, N, label, h_phiprod)
            #likelihood = likelihood + self.cal_locallikelihood(phi, nphi, Elogthetad, Elogbetad, N, label, h_phiprod)
            
        likelihood = likelihood + self.cal_globallikelihood(gamma, Elogtheta, batchD)
        
        #sstats = sstats * self._expElogbeta
        
        return (sstats, grad_mu, likelihood)
                
    def cal_locallikelihood(self, phi, cts, Elogthetad, Elogbetad, N, label, h_phiprod):
        likelihood = 0.0
        nphi = (phi.T * cts).T
        Elogpz_qz = n.sum(n.sum(nphi * (Elogthetad - n.log(phi))))
        Elogpw = n.sum(n.sum(nphi * Elogbetad.T))
        Elogpy = n.dot((1/N) * self._mu[label,:], n.average(nphi, axis = 0)) \
        - n.log(h_phiprod)
        likelihood = Elogpz_qz + Elogpw + Elogpy
            
        return likelihood  
    
    def cal_globallikelihood(self, gamma, Elogtheta, batchD):
        likelihood = 0.0
        likelihood += n.sum((self._alpha - gamma)*Elogtheta)
        likelihood += n.sum(gammaln(gamma) - gammaln(self._alpha))
        likelihood += sum(gammaln(self._alpha*self._K) - gammaln(n.sum(gamma, 1)))
        
        #likelihood = likelihood * self._D / batchD
        
        likelihood = likelihood + n.sum((self._eta-self._lambda)*self._Elogbeta)
        likelihood = likelihood + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        likelihood = likelihood + n.sum(gammaln(self._eta*self._V) - 
                              gammaln(n.sum(self._lambda, 1)))  
        
        return likelihood      
                  
    def calculatesfaux(self, phi, expmu, cts):
        #sf_aux = n.zeros((self._C, len(nphi)))
        #sf_aux_prod = n.ones(self._C)
        #sf_aux = n.dot(expmu, nphi.T)
        sf_aux = n.dot(expmu, phi.T)
        
        
        #sf_aux_power = n.zeros(sf_aux.shape)
        sf_aux_power = n.power(sf_aux, cts)
        
             
        
        sf_aux_prod = n.prod(sf_aux_power, axis = 1) + 1e-100
        h_phiprod = n.sum(sf_aux_prod)
        h = n.zeros((phi.shape))
        temp = (sf_aux_prod[:,n.newaxis] / sf_aux)
        for v in range(0, len(h)):
            #hvc = temp[:,v][:,n.newaxis] * n.power(expmu, cts[v])
            hvc = temp[:,v][:,n.newaxis] * expmu #* cts[v]
            hv = n.sum(hvc, axis = 0)
            h[v,:] = hv 
        
        return (h_phiprod, h)
            
    def calgradmu(self, phi, expmu, cts, label):
        gra_mu = n.zeros(expmu.shape)
        nphi = (phi.T * cts).T
        avephi = n.average(nphi, axis = 0)
        gra_mu[label,:] = avephi
        N = float(n.sum(cts))
        #sf_aux = n.zeros((self._C, len(cts)))
        #sf_aux_prod = n.ones(self._C)
        sf_aux = n.dot(expmu, phi.T)
        sf_aux_power = n.power(sf_aux, cts)
 
        sf_aux_prod = n.prod(sf_aux_power, axis = 1) +1e-100
        kappa_1 = 1.0 / n.sum(sf_aux_prod)
       
        sf_pra = n.zeros((self._C, self._K))
        
        temp = (sf_aux_prod[:,n.newaxis] / sf_aux)
        for c in range (0, self._C):
            temp1 = n.outer(temp[c,:], (1.0/N) * expmu[c,:])
            temp1 = temp1 * nphi
            sf_pra[c,:] = n.sum(temp1, axis = 0)
        
        
       
        sf_pra = sf_pra * (-1) * kappa_1
        gra_mu = gra_mu + sf_pra
        return gra_mu
        #sf_aux_prod[c] = sf_aux_prod[c] * sf_aux[c][v]
        
    def do_m_step(self, docs, doclabels):
        (wordids, wordcts, labels)=parse_document.parse_docs(docs, doclabels)
        (sstats, grad_mu, likelihood) = self.do_e_step(wordids, wordcts, labels)
        
        grad_lambda = -self._lambda + self._eta + (self._D/len(docs)) * sstats
        rho1 = n.power((self._iterations + self._tau1), -self._kappa)
        self._lambda = self._lambda + rho1 * grad_lambda
        #rho2 = n.power((self._iterations + self._tau2), -1)
        rho2 = 0.1
        self._mu = self._mu + rho2* grad_mu
        
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        
        self._iterations += 1 
        
        return likelihood
                
    def saveparameters(self):
        n.savetxt("lambda-%d.txt"%self._iterations, self._lambda)
        n.savetxt("mu-%d.txt"%self._iterations, self._mu)    
        
        
  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
