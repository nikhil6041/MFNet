import torch
from torch.nn.modules.distance import PairwiseDistance


class TripletLoss(torch.nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)     # distance b/w anchor image and positive image
        neg_dist = self.pdist.forward(anchor, negative)     # distance b/w anchor image and negative image

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)
        return loss

class QuadTripletLoss(torch.nn.Module):

    def __init__(self, alpha1,alpha2,alpha3,alpha4):
        
        super(QuadTripletLoss, self).__init__()
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        self.l1 = TripletLoss(alpha1)
        self.l2 = TripletLoss(alpha2)
        self.l3 = TripletLoss(alpha3)
        self.l4 = TripletLoss(alpha4)

    def forward(self, anchor, positive, negative, anchor_masked, positive_masked, negative_masked):
        
        pos_dist_u_u = self.pdist.forward(anchor, positive)     # distance b/w anchor image and positive image
        neg_dist_u_u = self.pdist.forward(anchor, negative)     # distance b/w anchor image and negaive image

        hinge_dist_u_u = torch.clamp(self.alpha + pos_dist_u_u - neg_dist_u_u, min=0.0)
        
        pos_dist_u_m = self.pdist.forward(anchor, positive_masked)  # distance b/w anchor image and positive masked image
        neg_dist_u_m = self.pdist.forward(anchor, negative_masked)  # distance b/w anchor image and negative masked image
        hinge_dist_u_m = torch.clamp(self.margin + pos_dist_u_m - neg_dist_u_m, min=0.0)
        
        pos_dist_m_u = self.pdist.forward(anchor_masked, positive)  # distance b/w anchor masked image and positive image
        neg_dist_m_u = self.pdist.forward(anchor_masked, negative)  # distance b/w anchor masked image and negative image
        hinge_dist_m_u = torch.clamp(self.margin + pos_dist_m_u - neg_dist_m_u, min=0.0)
        
        pos_dist_m_m = self.pdist.forward(anchor_masked, positive_masked)   # distance b/w anchor masked image and positive  masked image
        neg_dist_m_m = self.pdist.forward(anchor_masked, negative_masked)   # distance b/w anchor masked image and negative  masked image
        hinge_dist_m_m = torch.clamp(self.margin + pos_dist_m_m - neg_dist_m_m, min=0.0)

        loss_u_u = torch.mean(hinge_dist_u_u)   
        loss_u_m = torch.mean(hinge_dist_u_m)
        loss_m_u = torch.mean(hinge_dist_m_u)
        loss_m_m = torch.mean(hinge_dist_m_m)

        loss = loss_u_u + loss_u_m + loss_m_u + loss_m_m

        return loss
