#!/usr/bin/env python
"""
DESCRIPTION

    Identify the peaks in a (sequence) of image arrays.

EXAMPLES

    TODO: Show some examples of how to use this script.
"""

import sys, os, textwrap, traceback, argparse
import time
import shutil
#from pexpect import run, spawn
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.morphology import reconstruction


__version__ = 0.1


def guess_format(filename):
    """
    Try to guess the format of the filename based on its extension or
    characteristic name.
    """
    path, basename = os.path.split(filename)
    basename, ext = os.path.splitext(basename)
    if ext.lower() == '.mat':
        return 'MATLAB'
    else:
        raise ValueError('Could not deduce file format for {}'.format(filename))
#end guess_format


def guess_MATLAB_key(filename):
    """
    Try to guess from the contents of the file the key that accesses the
    eta-omega image data.
    """
    data = sio.loadmat(filename)
    key = [k for k in data.keys() if not k.startswith('__')]
    if len(key) == 1:
        return key[0]
    else:
        raise AttributeError('{} contains more than one key.'.format(filename))
#end guess_MATLAB_key


def normalized(arr):
    """
    Normalize the data in array, i.e. all values lie in the range
    [0, 1] except in the following cases:

      - If *arr* contains only one element, return ```ones_like(arr)```
      - If all elements in *arr* are the same (as determined by
        ```close(max(arr), min(arr))```, return ```zeros_like(arr)```.
    """
    arr = np.asarray(arr)
    if arr.size == 1:
        return np.ones_like(arr)
    elif np.isclose(np.max(arr), np.min(arr)):
        return np.zeros_like(arr)
    else:
        m,M = np.min(arr), np.max(arr)
        return (arr - m)/(M - m)
#end normalize


def threshold_masks(arr, val, direction='lt', relative=False):
    """
    Identify the connected components that are:

      - less than (direction = 'lt', default)
      - less-than-or-equal-to (direction = 'le')
      - greater than (direction = 'gt')
      - greater-than-or-equal-to (direction = 'ge')
      - equal (direction = 'eq')

    and can be taken *relative* to the domain. If *relative* is
    true, then *val* is taken to be a fraction of the range, between
    min(arr) and max(arr).
    """
    # threshold
    if relative:
        threshold = val*np.max(arr) + (1.-val)*np.min(arr)
    else:
        threshold = val
    # operation
    if direction == 'lt':
        op = lambda x,y: x < y
    elif direction == 'le':
        op = lambda x,y: x <= y
    elif direction == 'gt':
        op = lambda x,y: x > y
    elif direction == 'ge':
        op = lambda x,y: x >= y
    elif direction == 'eq':
        op = lambda x,y: x == y
    else:
        raise ValueError('Unrecognized operation to threshold_masks')
    # create mask
    mask = op(arr, threshold)
    # label
    labels, nlabels = measure.label(mask,
                                    background=0,
                                    return_num=True,
                                    connectivity=2)
    # create label masks
    lmasks = [(labels == i) for i in xrange(nlabels)]
    # return label masks
    return lmasks
#end threshold_masks


def clean(image, threshold=0.1, niter=20, selem=np.ones((3,3))):
    """
    Clean the image through a series of steps:
    
      1. normalize
      2. erode
      3. threshold (trim)
      4. measure (connected components)
    
    Parameters
    ----------
    :image, 2D nd.array: 2D array of intensity data
    :threshold, float or iterable of floats: values to use for identifying
        peaks. All values should be in the range (0, 1), exclusive. If a
        single float is specified, *niter* should be specified.
    :niter, int: Number of times *threshold* should be applied. Default: 20.
        This has no effect if *threshold* is an iterable.
    :selem, 2D array: mask to use for the erosion step
    
    Returns
    -------
    The cleaned image: a 2D nd.array the same size/shape as the original.
    """
    filtered = normalized(image)
    try:
        niter = len(threshold)
    except TypeError:
        threshold = niter*[threshold]
    for trim in threshold:
        # erosion
        seed = np.copy(filtered)
        seed[:-1, :-1] = filtered.max()
        filtered = reconstruction(seed, filtered, method='erosion', selem=selem)
        # threshold
        filtered -= trim
        # identify the remaining connected regions and normalize each separately
        labelMasks = threshold_masks(filtered, 0.0, direction='gt')
        for m in labelMasks:
            filtered[m] = normalized(filtered[m])
    return filtered
# end clean


def read_mat(filename, key):
    """
    Read a Matlab .mat file, returning the list of intensity maps
    stored in *key*.
    """
    data = sio.loadmat(filename)
    return data[key]
#end read_mat


def write_image(filename, arr, pts=None, minsize=None, **kwds):
    global args
    # imshow places x on vertical axis and y on horizontal
    ydom, xdom = arr.shape
    #set the size
    if minsize is not None:
        maxsize = 2**15/args.resolution
        size = np.array([xdom, ydom], dtype=float)/np.min((xdom, ydom))
        size *= minsize
        if np.max(size) > maxsize:
            size *= float(maxsize)/np.max(size)
        xsize, ysize = size
    else:
        xsize, ysize = np.array([xdom, ydom], dtype=float)/args.resolution
    fig = plt.figure(figsize=(xsize, ysize), dpi=args.resolution)
    # plot the image
    ax = fig.gca()
    kwds['interpolation'] = kwds.get('interpolation', 'none')
    ax.imshow(arr, **kwds)
    ax.set_xlabel(r'$\omega$', fontsize='large')
    ax.set_ylabel(r'$\eta$', fontsize='large')
    # plot any points
    if pts is not None:
        pts = np.asarray(pts)
        ax.plot(pts[:,1], pts[:,0], 'go', markersize=3)
        # resize (since adding points often adds padding)
        ax.set_xlim(0, xdom)
        ax.set_ylim(0, ydom)
    fig.savefig(filename, bbox_inches='tight', pad_inches=1./3.)
    fig.clf()
    plt.close()
#end write_image


def write_peak_pos(filename, positions):
    with open(filename, 'w') as ofs:
        for x,y in positions:
            ofs.write('{:d} {:d}\n'.format(x, y))
#end write_peak_pos


def ensure_square(img):
    """
    Ensure *img* is square.
    """
    if img.shape[0] != img.shape[1]:
        N = np.max(img.shape)
        di, dj = N - np.array(img.shape)
        if di > 0:
            # excess of columns: add rows
            return np.concatenate((img, np.zeros((di, img.shape[1]))),
                                  axis=0)
        else:
            # excess of rows: add columns
            return np.concatenate((img, np.zeros((img.shape[0], dj))),
                                  axis=1)
#end ensure_square


def gaussian_convolution(arr, sigma=1.0, width=None):
    """
    Convolve a 1D-array with a (unnormalized) gaussian. The gaussian
    is extended to *width* elements. *arr* is assumed to be equally
    spaced data.
    """
    width = int(10*sigma) if width is None else width
    if width < 1:
        raise ValueError('Invalid gaussian kernel width')
    kernel = np.exp(-0.5*(np.arange(-width, width+1)/sigma)**2)
    return np.convolve(arr, kernel, mode='same')
#end gaussian_convolution


def process_commandline():
    process_program_name()
    process_filelist()
    process_background_noise_threshold()
    process_colormap()
    process_input_formats()
    process_output_options()
    process_frame_number()
    process_treeline()
    process_peak_search()
    process_search_axis()
    process_sigma()
    process_resolution()

def process_program_name():
    global args
    # --- print version information --- #
    if args.verbose > 0:
        path, util = os.path.split(sys.argv[0])
        util, ext = os.path.splitext(util)
        sys.stdout.write('<prog version={}>{}'\
                         '<\prog>\n'.format(__version__, util))

def process_filelist():
    global args
    if args.verbose > 0:
        sys.stdout.write('<input type="vector">\n')
        for ifile in args.filelist:
            sys.stdout.write('  {}\n'.format(ifile))
        sys.stdout.write('</input>\n')

def process_background_noise_threshold():
    global args
    if args.verbose > 0:
        sys.stdout.write('<option value={}>background noise threshold' \
                         '</option>\n'.format(args.background))

def process_colormap():
    global args
    if args.verbose > 0:
        sys.stdout.write('<option value={}>color map' \
                         '</option>\n'.format(args.cmap))

def process_input_formats():
    global args
    # input file reader
    inputFormat = getattr(args, 'infun', guess_format(args.filelist[0]))
    if inputFormat == 'MATLAB':
        try:
            if args.iopt == "?":
                raise AttributeError()
        except AttributeError:
            try:
                args.iopt = guess_MATLAB_key(args.filelist[0])
            except AttributeError:
                sys.stdout.write('Specify with the --mat option which key' \
                                 'accesses the eta-omega image information:\n')
                data = sio.loadmat(ifile)
                for key in [k for k in data.keys() if not k.startswith('__')]:
                    sys.stdout.write('  {}\n'.format(key))
                raise
        args.infun = lambda ifile: read_mat(ifile, key=args.iopt)
        if args.verbose > 0:
            sys.stdout.write('<option value={}>input format' \
                             '</option>\n'.format(inputFormat))
            sys.stdout.write('<option value={}>input options' \
                             '</option>\n'.format(args.iopt))
    else:
        raise ValueError('Input format ({}) is not ' \
                         'recognized.'.format(inputFormat))

def process_output_options():
    global args
    if args.output is None:
        path, basename = os.path.split(args.filelist[0])
        basename, ext = os.path.splitext(basename)
        args.output = basename
        if args.verbose > 0:
            sys.stdout.write('<option value={}>output prefix' \
                             '</option>\n'.format(args.output))

def process_frame_number():
    global args
    # over what indices should we operate?
    if args.verbose > 0:
        sys.stdout.write('<option value={}>frame index' \
                         '</option>\n'.format(args.include))
    try:
        idx = int(args.include)
        args.include = lambda vec: [idx]
    except ValueError:
        if args.include == 'all':
            args.include = lambda vec: xrange(len(vec))
        else:
            raise ValueError('Unrecognized include action: ' \
                             '{}'.format(args.include))

def process_treeline():
    global args
    # validate numerical options
    if not 0.0 < args.treeline < 1.0:
        raise ValueError('Treeline must be set between 0.0 and 1.0, ' \
                         'exclusive.')
    if args.verbose > 0:
        sys.stdout.write('<option value={}>treeline' \
                         '</option>\n'.format(args.treeline))

def process_peak_search():
    global args
    # set the default steps
    if args.peakstep is None:
        args.peakstep = [0.1]

    # set the number of times these peak steps should be applied
    if args.peakreps < 1:
        raise ValueError('A positive number of repetitions must be ' \
                         'specified.')
    if args.verbose > 0:
        sys.stdout.write('<option value={}x{}>peak search' \
                         '</option>\n'.format(args.peakreps, args.peakstep))

def process_search_axis():
    global args
    if args.verbose > 0:
        sys.stdout.write('<option value={}>search axis' \
                         '</option>\n'.format(args.searchAlongAxis))

def process_sigma():
    global args
    if args.verbose > 0:
        sys.stdout.write('<option value={}>sigma' \
                         '</option>\n'.format(args.sigma))

def process_resolution():
    global args
    if args.verbose > 0:
        sys.stdout.write('<option value={}>resolution' \
                         '</option>\n'.format(args.resolution))



def read_data():
    # --- read data --- #
    timer = time.clock()
    data = []
    for ifile in args.filelist:
        data.extend(args.infun(ifile))
    timer = time.clock() - timer

    if args.verbose > 0:
        w, h = data[0].shape
        sys.stdout.write('<frames width={}, height={}, count={}, elapsed={}>' \
                         'read</frames>\n'.format(w, h, len(data), timer))

    return data
#end read_data    


def square_frames(data):
    timer = time.clock()
    for idx in args.include(data):
        data[idx] = ensure_square(data[idx])
    timer = time.clock() - timer

    if args.verbose > 0:
        w, h = data[0].shape
        sys.stdout.write('<frames width={}, height={}, elapsed={}>' \
                         'ensure square</frames>\n'.format(w, h, timer))
#end square_frames


def mask_frames(data):
    timer = time.clock()
    frameMasks = [gaussian_filter(frame, sigma=args.sigma) > args.background
                  for frame in data]
    timer = time.clock() - timer

    if args.verbose > 0:
        sys.stdout.write('<frames elapsed={}>mask gaussian convolution' \
                         '</frames>\n'.format(timer))

    return frameMasks
#end mask_frames


def multiply_intensity_by_negative_laplacian(data):
    timer = time.clock()
    for idx in args.include(data):
        frame = data[idx]
        Nx, Ny = frame.shape
        if args.searchAlongAxis in ('x', 'xy'):
            for ix in xrange(Nx):
                x = gaussian_convolution(frame[ix, :], args.sigma)
                xpp = np.gradient(np.gradient(x))
                frame[idx, :] = x*normalized(-xpp)
        if args.searchAlongAxis in ('y', 'xy'):
            for iy in xrange(Ny):
                y = gaussian_convolution(frame[:, iy], args.sigma)
                ypp = np.gradient(np.gradient(y))
                frame[:, idx] = y*normalized(-ypp)
        data[idx] = normalized(frame)
    timer = time.clock() - timer

    if args.verbose > 0:
        sys.stdout.write('<frames elapsed={}>I*normalized(-Laplacian(I))' \
                         '</frames>\n'.format(timer))
#end multiply_intensity_by_negative_laplacian


def optimized_frames(data, frameMasks):
    timer = time.clock()
    filtered = []
    for idx in args.include(data):
        frame = data[idx]
        mask = frameMasks[idx]
        optim = clean(frame*mask, threshold=args.peakreps*args.peakstep)
        filtered.append(optim)
    timer = time.clock() - timer

    if args.verbose > 0:
        sys.stdout.write('<frames elapsed={}>peaks cleaned' \
                         '</frames>\n'.format(timer))
        
    return filtered
#end optimized_frames


def peak_neighborhoods(data, frameMasks):
    timer = time.clock()
    maskLabels = []
    for idx in args.include(data):
        frame = data[idx]
        mask = frameMasks[idx]
        maskLabels.append(threshold_masks(mask, 0.5, direction='gt'))
    timer = time.clock() - timer

    if args.verbose > 0:
        sys.stdout.write('<frames elapsed={}>peak neighborhoods' \
                         '</frames>\n'.format(timer))

    return maskLabels
#end peak_neighborhoods


def peak_positions(data, filtered, maskLabels):
    """
    Returns a list of peak positions ((x0, y0), (x1, y1), ..., (xN, yN)).
    """
    timer = time.clock()
    peakPos = []
    for idx in args.include(data):
        masks = maskLabels[idx]
        frame = data[idx]
        optim = filtered[idx]
        index = np.reshape(np.arange(optim.size), optim.shape)
        xpeaks = []
        ypeaks = []
        # create a directory in which to store the features of this frame
        dstdir = "frame{:03d}".format(idx)
        featureno = 0
        try:
            # if dstdir already exists, then throw and OSError
            os.mkdir(dstdir)
        except OSError:
            pass
        # these are the masks for each feature (peak) neighborhood
        for m in masks:
            # spatial limits of this feature
            xy = np.where(m)
            xlo, xhi = np.min(xy[0]), np.max(xy[0])+1
            ylo, yhi = np.min(xy[1]), np.max(xy[1])+1
            # convenience: slice of frame and filtered image
            subframe = frame[xlo:xhi, ylo:yhi]
            suboptim = optim[xlo:xhi, ylo:yhi]
            # 2D array of indices to access this feature
            subindex = index[xlo:xhi, ylo:yhi]
            # get peak positions from filtered subframe (suboptim)
            peakMasks = threshold_masks(suboptim, args.treeline,
                                        direction='gt', relative=True)
            for pm in peakMasks:
                # local maximum (peak)
                ixy = np.argmax(suboptim[pm])
                # position of peak (flattened index)
                pos = (subindex[pm])[ixy]
                # (x,y) position in index where position occurs
                ix, iy = np.argwhere(index == pos)[0]
                xpeaks.append(ix)
                ypeaks.append(iy)
            # save this feature
            filename = "{}/feature{:04}.png".format(dstdir, featureno)
            featureno += 1
            pts = np.array(zip(xpeaks, ypeaks)) - np.array([xlo, ylo])
            write_image(filename, subframe, pts=pts, minsize=20/3.)
        # store the (x, y) peak positions
        peakPos.append(zip(xpeaks, ypeaks))
        # write the peak positions
        prefix = '{}-{:03d}'.format(args.output, idx)
        ofile = '{}.txt'.format(prefix)
        write_peak_pos(ofile, zip(xpeaks, ypeaks))
        # write the filtered image, including peak positions
        ofile = '{}.png'.format(prefix)
        write_image(ofile, optim, pts=peakPos[idx], cmap=args.cmap)
    timer = time.clock() - timer

    if args.verbose > 0:
        sys.stdout.write('<frames elapsed={}>peak positions' \
                         '</frames>\n'.format(timer))

    return peakPos
#end peak_positions


def main ():
    global args

    process_commandline()

    data = read_data()

    square_frames(data)

    frameMasks = mask_frames(data)

    multiply_intensity_by_negative_laplacian(data)

    # --- clean up the frames --- #
    filtered = optimized_frames(data, frameMasks)

    # --- identify each peak neighborhood in turn --- #
    maskLabels = peak_neighborhoods(data, frameMasks)

    # --- for each image and each peak neighborhood within that image
    # --- identify the peaks --- #
    peakPos = peak_positions(data, filtered, maskLabels)
#end 'def main ():'


if __name__ == '__main__':
    class MatAction(argparse.Action):
        def __init__(self, *args, **kwds):
            super(MatAction, self).__init__(*args, **kwds)
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, 'self.infun', 'MATLAB')
            setattr(namespace, 'self.inopt', values)
    try:
        start_time = time.time()
        parser = argparse.ArgumentParser(
                #prog='HELLOWORLD', # default: sys.argv[0], uncomment to customize
                description=textwrap.dedent(globals()['__doc__']),
                epilog=textwrap.dedent("""\
                    EXIT STATUS

                        0 on success

                    AUTHOR

                        Branden Kappes <bkappes@mines.edu>

                    LICENSE

                        This script is in the public domain, free from copyrights
                        or restrictions.
                        """))
        # positional parameters
        parser.add_argument('filelist',
                            metavar='file',
                            type=str,
                            nargs='+', # at least one filename
                            #nargs=argparse.REMAINDER, # if there are
                            help='File(s) containing a list of eta-omega ' \
                            'data. If more than one file is specified, all ' \
                            'must be of the same type, e.g. all MATLAB files ' \
                            'with the same key for accessing the eta-omega ' \
                            'data.')
        # optional parameters
        parser.add_argument('--background-noise-threshold',
                            dest='background',
                            action='store',
                            type=float,
                            default=1.0e-6,
                            help='Maximum noise that will be encountered ' \
                            'in the background and that should be ignored. ' \
                            'Default: 1.e-6.')
        parser.add_argument('--cmap',
                            dest='cmap',
                            action='store',
                            type=str,
                            default='gist_stern',
                            help='Set the colormap used to generate the ' \
                            'image maps. Default: gist_stern.')
        parser.add_argument('--include-only',
                            metavar='INDEX',
                            dest='include',
                            action='store',
                            default='all',
                            help='Specify the index of a single frame. The ' \
                            'special case "all" includes all frames. ' \
                            'Default: "all".')
        # MatAction sets ifmt='MATLAB' and iopt to the argument passed to
        # this argument
        parser.add_argument('--mat',
                            metavar='KEY',
                            action=MatAction,
                            default="?",
                            help='Input is formatted as a MATLAB .mat file. ' \
                            'The argument to this flag sets the KEY used to ' \
                            'access the list of image information. (The data ' \
                            'read from a .mat file is accessed as an ' \
                            'associative map.) Each frame (image) is a 2D ' \
                            'matrix of intensity values at uniform steps in ' \
                            'eta (row) and omega (column).')
        parser.add_argument('-o',
                            '--output-prefix',
                            dest='output',
                            action='store',
                            type=str,
                            default=None,
                            help='Set the output file prefix. Default: ' \
                            'determined automatically from input file.')
        parser.add_argument('--pr', '--peak-reps',
                            dest='peakreps',
                            action='store',
                            type=int,
                            default=20,
                            help='Number of times to repeat the peak search ' \
                            'steps constructed by calls to ' \
                            '"--peak-stepsize". Default: 20.')
        parser.add_argument('--ps', '--peak-stepsize',
                            dest='peakstep',
                            action='append',
                            type=float,
                            default=None,
                            help='Add another peak step, i.e. drop data ' \
                            'whose normalized intensity falls below PEAKSTEP ' \
                            'and renormalize. This argument can be specified ' \
                            'multiple times, and the resulting list repeated ' \
                            'PEAKREP times. (See "--peak-reps".) Default: 0.1')
        parser.add_argument('--resolution',
                            dest='resolution',
                            type=int,
                            default=300,
                            help='Set the output image resolution, in DPI ' \
                            '(dots per inch). Default: 300.')
        # mutually exclusive group
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--search-along-x',
                           dest='searchAlongAxis',
                           action='store_const',
                           const='x',
                           help='Search for peaks along the x-axis.')
        group.add_argument('--search-along-xy',
                           dest='searchAlongAxis',
                           action='store_const',
                           const='xy',
                           help='Search for peaks along both x and y axes.')
        group.add_argument('--search-along-y',
                           dest='searchAlongAxis',
                           action='store_const',
                           const='y',
                           help='Search for peaks along the y-axis.')
        parser.set_defaults(searchAlongAxis='y')
        #end group
        parser.add_argument('--sigma',
                            dest='sigma',
                            action='store',
                            type=float,
                            default=1.0,
                            help='Gaussian smoothing is used to identify ' \
                            'peak neighborhoods and exaggerate the ' \
                            'difference between the peaks and troughs. ' \
                            'Default: 1.0 (units = pixels).')
        parser.add_argument('--tree-line',
                            dest='treeline',
                            action='store',
                            type=float,
                            default=0.7,
                            help='Only the peak is above tree line, i.e. ' \
                            'once intensities have been processed, then ' \
                            'use this value as a floor (relative to the ' \
                            'local maximum) to further isolate the peak. ' \
                            'Acceptable values between 0. and 1., exclusive. '
                            'Default: 0.7.')
        parser.add_argument('-v',
            '--verbose',
            action='count',
            default=0,
            help='Verbose output')
        parser.add_argument('--version',
            action='version',
            version='%(prog)s {}'.format(__version__))
        args = parser.parse_args()
        # check for correct number of positional parameters
        #if len(args.filelist) < 1:
            #parser.error('missing argument')
        # timing
        if args.verbose > 0: print time.asctime()
        main()
        if args.verbose > 0: print time.asctime()
        if args.verbose:
            delta_time = time.time() - start_time
            hh = int(delta_time/3600.); delta_time -= float(hh)*3600.
            mm = int(delta_time/60.); delta_time -= float(mm)*60.
            ss = delta_time
            print 'TOTAL TIME: {0:02d}:{1:02d}:{2:06.3f}'.format(hh,mm,ss)
        sys.exit(0)
    except KeyboardInterrupt, e: # Ctrl-C
        raise e
    except SystemExit, e: # sys.exit()
        raise e
    except Exception, e:
        print 'ERROR, UNEXPECTED EXCEPTION'
        print str(e)
        traceback.print_exc()
        os._exit(1)
#end 'if __name__ == '__main__':'
