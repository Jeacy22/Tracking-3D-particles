import os
import torch
from models.centernet_training import focal_loss,reg_l1_loss
from utils1.utils import get_lr
from tqdm import tqdm
import numpy as np




def fit_one_epoch(model_train,model,writer,optimizer,epoch,epoch_step,epoch_step_val,gen,gen_val,
                  Epoch,cuda,fp16,scaler,save_dir,local_rank=0,num_classes=2):
    total_r_loss = 0
    total_c_loss = 0
    total_h_loss = 0
    total_offset_loss = 0
    total_loss=0
    val_loss=0
    val_depth_loss = 0
    val_off_loss = 0
    val_c_loss = 0
    val_pos_loss = 0
    min_loss = 1


    if local_rank ==0:
        print('start train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.train()

    for iteration,batch in enumerate(gen):
        if iteration>=epoch_step:
            break
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
        batch_images, batch_hms, batch_depths, batch_regs, batch_reg_masks = batch

        optimizer.zero_grad()
        if not fp16:
            hm,depths,offset = model_train(batch_images)
            c_loss = focal_loss(hm,batch_hms)
            h_loss = reg_l1_loss(depths,batch_depths,batch_reg_masks)
            off_loss = reg_l1_loss(offset,batch_regs,batch_reg_masks)

            loss = c_loss+h_loss+off_loss

            total_loss += loss.item()
            total_c_loss += c_loss.item()
            total_r_loss += h_loss.item()+off_loss.item()
            total_h_loss += h_loss.item()
            total_offset_loss += off_loss.item()

            loss.backward()
            optimizer.step()

        else:
            from torch.cuda.amp import autocast
            with autocast():

                hm,depths,offset = model_train(batch_images)
                c_loss = focal_loss(hm,batch_hms)
                h_loss = reg_l1_loss(depths,batch_depths,batch_reg_masks)
                off_loss = reg_l1_loss(offset,batch_regs,batch_reg_masks)

                loss = c_loss+h_loss+off_loss

                total_loss  += loss.item()
                total_c_loss += c_loss.item()
                total_r_loss += h_loss.item() + off_loss.item()
                total_h_loss += h_loss.item()
                total_offset_loss += off_loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if local_rank==0:
            pbar.set_postfix(**{'pos_loss':total_r_loss/(iteration+1),
                                'c_loss':total_c_loss/(iteration+1),
                                'depth_loss':total_h_loss/(iteration+1),
                                'offset_loss':total_offset_loss/(iteration+1),
                                'total_loss':total_loss/(iteration+1),
                                'lr':get_lr(optimizer)})
            pbar.update(1)

    writer.add_scalar('pos_loss', total_r_loss/epoch_step, epoch+1)
    writer.add_scalar("c_loss", total_c_loss/epoch_step, epoch+1)
    writer.add_scalar('depth_loss', total_h_loss/epoch_step, epoch+1)
    writer.add_scalar('offset_loss', total_offset_loss/epoch_step, epoch+1)
    writer.add_scalar('total_loss', total_loss / epoch_step, epoch+1)
    writer.add_scalar('lr', get_lr(optimizer), epoch)


    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()

    for iteration,batch in enumerate(gen_val):
        if iteration >epoch_step_val:
            break
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
            batch_images, batch_hms, batch_depths, batch_regs, batch_reg_masks = batch
            hm, depths, offset = model_train(batch_images)
            c_loss = focal_loss(hm, batch_hms)
            h_loss = reg_l1_loss(depths, batch_depths, batch_reg_masks)
            off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

            loss            = c_loss + h_loss + off_loss
            val_loss        += loss.item()
            val_depth_loss += h_loss.item()
            val_off_loss += off_loss.item()
            val_c_loss += c_loss.item()
            val_pos_loss +=  c_loss.item()+off_loss.item()


            if local_rank == 0:
                pbar.set_postfix(**{'val_loss':val_loss/(iteration+1),
                                    'val_c_loss':val_c_loss/(iteration+1),
                                    'val_offset_loss':val_off_loss/(iteration+1),
                                    'val_depth_loss':val_depth_loss/(iteration+1)})
                pbar.update(1)


    pbar.close()
    print('Finish Validation')
    writer.add_scalar('val_loss', val_loss/epoch_step_val, 1+ epoch)
    writer.add_scalar('val_c_loss', val_c_loss/epoch_step_val, 1+ epoch)
    writer.add_scalar('val_offset_loss', val_off_loss/epoch_step_val, 1+ epoch)
    writer.add_scalar('val_depth_loss', val_depth_loss/epoch_step_val, 1+ epoch)
    writer.add_scalar('val_pos_loss', val_pos_loss / epoch_step_val, 1 + epoch)
    #eval_callback.on_epoch_end(epoch + 1, model_train)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

    now_loss = val_loss/epoch_step_val

    if epoch==200:
        torch.save(model.state_dict(), os.path.join(save_dir, "200_epoch_weights.pth"))



    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


















