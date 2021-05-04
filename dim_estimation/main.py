import os
import config
from utils import *
import csv
from argparse import ArgumentParser
import json


def main():
    print('~~~ Starting dimension estimation! ~~~')

    # load config file
    args = config.load_args()

    # make output folder if it doesn't exist already
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # save args to json file
    with open(args.save_dir + 'commandline_args.txt', 'w') as file2:
        json.dump(args.__dict__, file2, indent=2)

    # get model
    print(' > Loading model...')
    model = get_model(args)
    device = args.device
    model.cuda(device)
    model.eval()

    # get dataset
    print(' > Preparing dataset...')
    dataloader = get_dataloader(args)

    if args.model.split('_')[0] == 'vit':
        patch_size = int(args.model.split('_')[-2][-2:])

    # create dict with n_factor lists and factor list
    factor_list = []
    output_dict = {'example1': [],
                   'example2': []}

    print(' > Processing starting...')
    # for-loop inference and store values as numpy array
    for i, (factor, example1, example2, _, _) in enumerate(dataloader):

        # move data to GPU
        example1, example2 = example1.cuda(device), example2.cuda(device)

        # pass images through model and get distribution mean
        if args.model.split('_')[0] == 'vit':
            output1 = model(example1, patch=patch_size).mode()[0]
            output2 = model(example2, patch=patch_size).mode()[0]
        else:
            output1 = model(example1).mode()[0]
            output2 = model(example2).mode()[0]

        # add factor and output to list / array for processing dimensions later on
        factor_list.append(factor.detach().cpu().numpy())
        output_dict['example1'].append(output1.detach().cpu().numpy())
        output_dict['example2'].append(output2.detach().cpu().numpy())

        if i % 10000 == 0:
            print('Processing example {}/{}'.format(i, len(dataloader)))

    print(' > Finished processing examples...')
    print(' > Starting Dimensionality Estimation!')

    # dimensionality estimation
    dims, dims_percent = dim_est(output_dict, factor_list, args)


    print(" >>> Estimated factor dimensionalities: {}".format(dims))
    print(" >>> Ratio to total dimensions: {}".format(dims_percent))

    print('Saving results to {}'.format(args.save_dir))

    # save to output folder
    with open(args.save_dir + '/' + args.model + '_dim_est.csv', mode='w') as file1:
        writer = csv.writer(file1, delimiter=',')
        writer.writerow(dims)
        writer.writerow(dims_percent)


if __name__ == '__main__':
    main()
