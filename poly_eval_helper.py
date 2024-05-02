import numpy as np
import sklearn.metrics

def __convert_piano_roll_to_note_onset_duration(x):
    '''

    :param x: numpy, num_songs*num_frames(32)*MIDI_DIM(128) [0,1,2]={silence, sustain, onset}
    :return: numpy, num_songs*num_frames(32)*MIDI_DIM(128), value=duration if any onset otherwise 0
    '''
    num_songs=x.shape[0]
    num_frames=x.shape[1]
    midi_dim=x.shape[2]
    nk=int(np.floor(np.log2(num_frames-1)))+1
    duration_matrix=np.zeros_like(x,dtype=np.int32)
    duration_matrix[x==1]=1
    for log_d in range(nk):
        d=2**log_d
        shift_duration_matrix=np.zeros_like(duration_matrix,dtype=np.int32)
        shift_duration_matrix[:,:-d,:]=duration_matrix[:,d:,:]
        duration_matrix=(duration_matrix==d)*shift_duration_matrix+duration_matrix
    result=np.zeros_like(x,dtype=np.int32)
    result[:,:-1,:]=duration_matrix[:,1:,:]
    result[x==2]+=1
    result[x!=2]=0
    return result



def __note_level_reconstruction(original_x,recon_x):
    print('[NOTE-LEVEL RECONSTRUCTION]')
    original_notes=__convert_piano_roll_to_note_onset_duration(original_x)
    recon_notes=__convert_piano_roll_to_note_onset_duration(recon_x)
    num_positive=np.sum(recon_notes>0)
    num_true=np.sum(original_notes>0)
    tp_onset=np.sum(np.logical_and(recon_notes>0,original_notes>0))
    tp_offset=np.sum(np.logical_and(original_notes==recon_notes,original_notes>0))
    def report_binary_score(tp,name):
        p=tp/num_positive
        r=tp/num_true
        f1=2*p*r/(p+r)
        print('%s: \tP=%.4f,R=%.4f,F1=%.4f'%(name,p,r,f1))
    report_binary_score(tp_onset,'PITCH+ONSET')
    report_binary_score(tp_offset,'PITCH+ON+OFF')
    durations_recon=recon_notes[np.logical_and(recon_notes>0,original_notes>0)]
    durations_origin=original_notes[np.logical_and(recon_notes>0,original_notes>0)]
    durations_tp=np.minimum(durations_origin,durations_recon).sum()
    durations_p=durations_tp/np.sum(durations_recon)
    durations_r=durations_tp/np.sum(durations_origin)
    durations_f1=2*durations_p*durations_r/(durations_p+durations_r)
    print('%s: \tP=%.4f,R=%.4f,F1=%.4f'%('GUS DURATION',durations_p,durations_r,durations_f1))

def __unit_test():
    piano_roll=np.array([
        [[2,0,0,1,0],
         [2,1,0,1,1],
         [2,1,1,0,1],
         [2,1,1,1,0],
         [2,1,1,1,1],
         [0,2,0,0,1]],
        [[0,2,1,0,1],
         [1,2,1,1,0],
         [1,0,2,1,0],
         [2,0,2,1,0],
         [2,1,2,0,2],
         [2,2,2,2,1]],
        [[0,0,0,0,0],
         [0,1,1,1,0],
         [0,1,2,1,0],
         [0,1,1,2,2],
         [2,0,1,2,0],
         [0,0,0,0,2]]
    ]).transpose((0,2,1))
    print(__convert_piano_roll_to_note_onset_duration(piano_roll).transpose((0,2,1)))


def collect_statistics(original_x,recon_x,z_mu,z_logvar):
    '''
    Polyphonic VAE evaluator, version 2.0, by jjy
    :param original_x: numpy, num_songs*num_frames(32)*MIDI_DIM(128) [0,1,2]={silence, sustain, onset}
    :param recon_x: numpy, num_songs*num_frames(32)*MIDI_DIM(128) [0,1,2]={silence, sustain, onset}
    :param z_mu: numpy, num_songs*z_dim
    :param z_logvar: numpy, num_songs*z_dim
    '''
    def report_binary_score(p,r,f0,index,name):
        print('%s: \tP=%.4f,R=%.4f,F1=%.4f'%(name,p[index],r[index],f0[index]))
    def average_kld(z_mu,z_sigma):
        return (-0.5*np.mean(1+z_sigma-z_mu**2-np.exp(z_sigma)))
    original_x=np.array(original_x)
    recon_x=np.array(recon_x)
    z_mu=np.array(z_mu)
    z_logvar=np.array(z_logvar)
    flat_original_x=original_x.reshape((-1))
    flat_recon_x=recon_x.reshape((-1))
    p,r,f,_=sklearn.metrics.precision_recall_fscore_support(flat_original_x,flat_recon_x,labels=[0,1,2])
    __note_level_reconstruction(original_x,recon_x)
    print('[FRAME-LEVEL RECONSTRUCTION]')
    print('OVERALL ACC=%.4f'%((flat_original_x==flat_recon_x).sum()/len(flat_recon_x)))
    report_binary_score(p,r,f,0,'SILENCE')
    report_binary_score(p,r,f,1,'SUSTAIN')
    report_binary_score(p,r,f,2,'ONSET')
    print('[KLD]')
    print('Z_DIM=%d'%z_mu.shape[-1])
    print('AVERAGE KLD=%f'%average_kld(z_mu.reshape((-1)),z_logvar.reshape((-1))))


