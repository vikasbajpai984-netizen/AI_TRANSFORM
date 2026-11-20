import argparse 
import torch
from pathlib import Path
from utils.utils import *
from torch.utils.data import DataLoader
from utils.models import *
import torch.optim as optim
from tqdm import tqdm 
from torchvision.utils import save_image

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type = str, default = '/Users/vikasbajpai/Desktop/kamal/MAJOR PROJECTS/project-2/NST/content_data',
                        help = 'Location of content dataset')
    
    parser.add_argument('--style_dir', type = str, default = '/Users/vikasbajpai/Desktop/kamal/MAJOR PROJECTS/project-2/NST/style_data',
                        help = 'Location of style dataset')
    
    parser.add_argument('--vgg', type = str, default = '/Users/vikasbajpai/Desktop/kamal/MAJOR PROJECTS/project-2/NST/vgg_normalised.pth',
                        help = 'Location of vgg')
    
    parser.add_argument('--experiment', type = str, default = 'experiment1',
                        help = 'Name of experiment') 
    
    parser.add_argument('--final_size', type = int, default = 512,
                        help = 'Final size of images')  
     
    parser.add_argument('--content_size', type = int, default = 256,
                        help = 'Content size of images') 
    
    parser.add_argument('--style_size', type = int, default = 256,
                        help = 'Style size of images')
     
    parser.add_argument('--crop', action='store_true', default=True,
                        help = 'Whether to crop images')
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'Batch size for training')
    
    parser.add_argument('--lr', type=float, default = 1e-4, 
                        help = 'Learning rate for optimizer')
    
    parser.add_argument('--lr_decay', type=float, default = 5e-5,
                        help = 'Learning rate decay for optimizer')
    
    parser.add_argument('--epochs', type=int, default =1, 
                        help = 'Number of epochs for training')
    
    parser.add_argument('--content_weight', type=float, default=1.5,
                        help='Content weight')
    
    parser.add_argument('--style_weight', type=float, default=10.0,
                        help='Style weight')
    
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log interval')
    
    parser.add_argument('--save_interval', type=int, default=2,
                        help='Save interval')
    
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from checkpoint')
    
    parser.add_argument('--decoder_path', type=str, default=None,
                        help='Path to decoder checkpoint')  
    
    parser.add_argument('--optimizer_path', type=str, default=None,
                        help='Path to optimizer checkpoint')
    return parser.parse_args()

def main():
    args = parse_arguments()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path('experiment') /args.experiment
    save_dir.mkdir(exist_ok = True, parents = True)
    
    #saving the arguments to a text file 
    with open(save_dir / 'args.txt','w') as f:
        for key , value in vars(args).items():
            f.write(f'{key}: {value}\n')

    style_transform = get_transforms(args.content_size, args.crop, args.final_size)
    content_transform = get_transforms(args.style_size, args.crop, args.final_size)
    
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)
    
    content_loader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=True)
    style_loader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False, drop_last=True)
    
    
    # print(f'Content Loader size: {len(content_loader)}')
    # print(f'Style Loader size: {len(style_loader)}')
    
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)
    optimizer = optim.Adam(decoder.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: 1.0/(1.0 + args.lr_decay*epoch))
    
    print("Resume argument:", args.resume)

    if args.resume:
        decoder.load_state_dict(
            torch.load(args.decoder_path, map_location=device)
        )

        optimizer.load_state_dict(
            torch.load(args.optimizer_path, map_location=device)
        )

    print("Training...")
    
    # loss function
    MSE = torch.nn.MSELoss()
    running_loss = None
    running_closs = None
    running_sloss = None
    encoder.eval()
    
    start_epoch = 98 if args.resume else 0

    print(f"Starting epoch: {start_epoch + 1}")
    print(f"Total target epochs: {args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nNow training Epoch {epoch + 1}")
        #creating a proqress
        progress_bar = tqdm((zip(content_loader, style_loader)),
                            total = min(len(content_loader), len(style_loader)))
        running_loss = 0
        running_closs = 0
        running_sloss = 0
        
        for content_batch , style_batch in progress_bar:
            
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)
            
            # print(f'content_batch: {content_batch.shape}')
            # print(f'style_batch: {style_batch.shape}')

            #forward pass(passing the content and style images through the encoder)
            c_features = encoder(content_batch)
            s_features = encoder(style_batch)
            
            t = adaptive_instance_normalization(c_features[-1], s_features[-1])
            g = decoder(t)
            
            g_features = encoder(g)
            loss_c = MSE(g_features[-1], t) * args.content_weight

            loss_s = 0
            for g_f, s_f in zip(g_features, s_features):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s += MSE(g_mean, s_mean) + MSE(g_std, s_std)
            
            loss_s = loss_s * args.style_weight

            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Loss:{loss.item():4f}, Content Loss: {loss_c.item():4f}, Style Loss: {loss_s.item():4f}')

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
        
        scheduler.step()
        
        running_loss /= len(content_loader)
        running_closs /= len(content_loader)
        running_sloss /= len(content_loader)

        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Epoch {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_closs:4f}, Style Loss: {running_sloss:4f}')

        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)
        

if __name__ == '__main__':
    main()