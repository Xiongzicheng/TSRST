import os
from time import time
import argparse

from einops import rearrange, reduce, repeat
import numpy as np
import skimage
import torch
import cv2

from tqdm import tqdm
from utils import load_image
from torchvision import transforms
from utils import load_pickle, save_pickle
from image import smoothen
from gigapath.pipeline import load_tile_slide_encoder
from torch.utils.data import DataLoader, TensorDataset


def patchify(x, patch_size):
    shape_ori = np.array(x.shape[:2])
    shape_ext = (
            (shape_ori + patch_size - 1)
            // patch_size * patch_size)
    x = np.pad(
            x,
            (
                (0, shape_ext[0] - x.shape[0]),
                (0, shape_ext[1] - x.shape[1]),
                (0, 0)),
            mode='edge')
    tiles_shape = np.array(x.shape[:2]) // patch_size
    # x = rearrange(
    #         x, '(h1 h) (w1 w) c -> h1 w1 h w c',
    #         h=patch_size, w=patch_size)
    # x = rearrange(
    #         x, '(h1 h) (w1 w) c -> (h1 w1) h w c',s
    #         h=patch_size, w=patch_size)
    tiles = []
    coords = []
    for i0 in range(tiles_shape[0]):
        a0 = i0 * patch_size  # TODO: change to patch_size[0]
        b0 = a0 + patch_size  # TODO: change to patch_size[0]
        for i1 in range(tiles_shape[1]):
            a1 = i1 * patch_size  # TODO: change to patch_size[1]
            b1 = a1 + patch_size  # TODO: change to patch_size[1]
            tiles.append(x[a0:b0, a1:b1])
            coords.append(torch.from_numpy(np.array([a0, a1])).float())

    shapes = dict(
            original=shape_ori,
            padded=shape_ext,
            tiles=tiles_shape)
    return tiles, shapes, torch.stack(coords, dim=0)


def get_data(prefix):
    img = load_image(f'{prefix}image.jpg')
    return img

def get_embeddings_tiles(model, x):
    model = model.cuda()

    x = model.patch_embed(x)

    cls_token = model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)

    x = x + model.pos_embed
    x = model.pos_drop(x)

    for block in model.blocks:
        x = block(x)
    return x[:, :-1, :]

def get_embeddings_slide(model, x, coords):

    coords = coords.unsqueeze(1)

    model = model.cuda()
    model.eval()

    x = model.patch_embed(x)

    pos = model.coords_to_pos(coords, model.tile_size)

    x = x + model.pos_embed[:, pos, :].squeeze(0)

    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    x = model.encoder(src_tokens=None, token_embeddings=x)["encoder_out"]

    x = model.norm(x)

    return x[:, :-1, :]

def eval_transforms():
    """ """
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    eval_t = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    return eval_t

def get_embeddings(img, pretrained=True, device='cuda'):
    '''
    Extract embeddings from histology tiles
    Args:
        tiles: Histology image tiles.
            Shape: (N, H, W, C).
            `H` and `W` are both divisible by 256.
            Channels `C` include R, G, B, foreground mask.
    Returns:
        emb_cls: Embeddings of (256 x 256)-sized patches
            Shape: (H/256, W/256, 384)
        emb_sub: Embeddings of (16 x 16)-sized patches
            Shape: (H/16, W/16, 384)
    '''
    print('Extracting embeddings...')
    # t0 = time()

    tile_size = 224
    tiles, shapes, coords = patchify(img, patch_size=tile_size)

    torch_tiles = []
    for tile in tiles:
        x = tile.astype(np.float32) / 255.0
        x = eval_transforms()(x)
        torch_tiles.append(x)

    combined_tensor = torch.stack(torch_tiles, dim=0)

    dataset = TensorDataset(combined_tensor, coords)

    tile_dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    tile_encoder, slide_encoder = load_tile_slide_encoder(global_pool=False)

    slide_embeddings = []

    with torch.cuda.amp.autocast(dtype=torch.float16):
        with torch.no_grad():
            for batch_tensors, batch_coords in tqdm(tile_dataloader, desc='Processing tiles'):
                tile_emb = get_embeddings_tiles(tile_encoder, batch_tensors.cuda())
                slide_emb = get_embeddings_slide(slide_encoder, tile_emb.cuda(), batch_coords.cuda())
                slide_embeddings.append(slide_emb.detach().cpu())

    slide_embeddings = torch.cat(slide_embeddings)
    slide_embeddings = slide_embeddings.view(-1, 14, 14, 768)
    emb = rearrange(slide_embeddings,
                    '(h1 w1) h2 w2 c -> (h1 h2) (w1 w2) c',
                    h1=shapes['tiles'][0], w1=shapes['tiles'][1]
                    )
    subpatch_size = (16, 16)
    shape_orig = np.array(shapes['original']) // subpatch_size
    emb = emb[:shape_orig[0], :shape_orig[1]]

    return emb

def smoothen_embeddings(
        embs, size, kernel,
        method='cv', groups=None, device='cuda'):
    if groups is None:
        groups = embs.keys()
    out = {}
    for grp, em in embs.items():
        if grp in groups:
            if isinstance(em, list):
                smoothened = [
                        smoothen(
                            c[..., np.newaxis], size=size,
                            kernel=kernel, backend=method,
                            device=device)[..., 0]
                        for c in em]
            else:
                smoothened = smoothen(em, size, method, device=device)
        else:
            smoothened = em
        out[grp] = smoothened
    return out


def save_embeddings(x, outfile):
    print('Saving embeddings...')
    t0 = time()
    save_pickle(x, outfile)
    print(int(time() - t0), 'sec')
    print('Embeddings saved to', outfile)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--reduction-method', type=str, default=None)
    parser.add_argument('--n-components', type=float, default=None)
    parser.add_argument('--smoothen-method', type=str, default='cv') # default='cv'
    parser.add_argument('--random-weights', action='store_true')
    parser.add_argument('--use-cache', action='store_true')
    parser.add_argument('--no-shift', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    np.random.seed(0)
    torch.manual_seed(0)

    # load data
    wsi = get_data(prefix=args.prefix)

    if args.use_cache:
        cache_file = args.prefix + 'embeddings-hist-raw.pickle'
    if args.use_cache and os.path.exists(cache_file):
        embs = load_pickle(cache_file)
    else:
        # extract prov-gigapath embeddings
        embs = get_embeddings(
                wsi, pretrained=(not args.random_weights),
                device=args.device)
    if args.use_cache:
        save_embeddings(embs, cache_file)

    del wsi
    result = {}
    result['emb'] = [np.array(embs[:,:,i]) for i in range(embs.shape[2])]

    # smoothen embeddings
    if args.smoothen_method is not None:
        print('Smoothening embeddings...')
        t0 = time()
        result = smoothen_embeddings(
                result, size=16, kernel='uniform', groups=['emb'],
                method=args.smoothen_method,
                device=args.device)
        print('runtime:', int(time()-t0))


    save_embeddings(result, args.prefix + 'embeddings-hist.pickle')


if __name__ == '__main__':
    main()
