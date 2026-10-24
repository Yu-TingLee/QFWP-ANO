from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
)

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "Exchange": Dataset_Custom,
    "ILI": Dataset_Custom,
    "custom": Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1

    # Evaluation keeps the last partial batch so small splits are not emptied.
    is_train = flag == "train"
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
    )
    print(flag, len(data_set))

    drop_last = is_train
    if drop_last and len(data_set) < args.batch_size:
        print(
            f"  [data] {flag}: {len(data_set)} < batch_size {args.batch_size}; drop_last disabled"
        )
        drop_last = False
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
