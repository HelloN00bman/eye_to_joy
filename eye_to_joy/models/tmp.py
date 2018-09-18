def threshold_joy(tensor, cond):                                                           
    mask = cond(tensor)                                                                    
    right_mask = mask[:,:,:,0].expand(3,config.batch_size,config.sequence_length,config.future_length).permute(1,2,3,0)                          
    middle_mask = mask[:,:,:,1].expand(3,config.batch_size,config.sequence_length,config.future_length).permute(1,2,3,0)
    left_mask = mask[:,:,:,2].expand(3,config.batch_size,config.sequence_length,config.future_length).permute(1,2,3,0)
    thresh = right_mask + middle_mask + left_mask         
    return thresh  


def prep_label(label):
    thresh_up = threshold_joy(label, lambda x: x>1)
    thresh_down = threshold_joy(label, lambda x: x<-1)
    thresh_nan = threshold_joy(label, lambda x: torch.isnan(x))

    thresh = thresh_up + thresh_down + thresh_nan
    thresh[thresh>1] = 1

    pos_numel = thresh[:,:,:2].numel()
    pos_thresh = torch.sum(thresh[:,:,:2]) / 2
    pos_tot = pos_numel - pos_thresh

    mode_numel = thresh[:,:,2].numel()
    mode_thresh = torch.sum(thresh[:,:,2])
    mode_tot = mode_numel - mode_thresh

    label[thresh] = 0

    pos = label[:,:,:,:2]
    X = torch.from_numpy(np.digitize(pos, np.linspace(-1,1,16))-1).view(-1,2)
    func = lambda x,y: x*16+y #if x*16+y in config.pos_inds else 0
    # import IPython; IPython.embed()
    pos_labels = torch.from_numpy(np.array([config.ind_to_class[int(func(x,y))] for x,y in zip(X[:,0], X[:,1])]))

    mode = label[:,:,:,2]
    mode_labels = (mode>0).float()

    return {
        'pos': pos_labels,
        'mode': mode_labels,
        'pos_tot': pos_tot.data.cpu().numpy(), 
        'mode_tot': mode_tot.data.cpu().numpy(), 
        'pos_numel': pos_numel,
        'mode_numel': mode_numel,
    }

