
import tensorflow as tf

l = tf.keras.layers


def conv_1d(x, filters, kernel_size, 
            separable=True, use_bias=False, 
            activation='relu', padding='same'):
    conv_layer = l.SeparableConvolution1D if separable else l.Conv1D
    return conv_layer(filters, kernel_size, use_bias=use_bias, 
        activation=activation, padding=padding)(x)


def switch(x, **kwargs):
    if not has_len(x):
        v = kwargs.get(x, None)
    else:
        v = lmap(lambda e: kwargs.get(e, None), x)
    if any(empty(v)):
        print("Don't recognize key `{}`".format(x))
    return v

lmap = lambda f, x: list(map(f, x))

def conv_block_1d(inputs, 
                  filter_sizes, 
                  kernel_sizes, 
                  depth=None, 
                  separable=True, 
                  cap=None,
                  return_all=True):
    if cap is None:
        cap = l.GlobalMaxPooling1D
    if depth is None:
        depth = 2 if is_scalar(filter_sizes) else len(filter_sizes)

    filter_sizes = filter_sizes * depth
    kernel_sizes = kernel_sizes * depth 

    if isinstance(cap, str):
        if not nzchar(cap): # if not any nulls in array `cap`
            cap = None
        else:
            cap = switch(
                cap,
                max=l.MaxPooling1D(),
                avg=l.AvgPool1D(),
                batchnorm=l.BatchNormalization()                
            )

    top = inputs
    stack = [0] * (depth * len(cap))

    for i in range(depth):
        stack[i] = top = conv_1d(top, filter_sizes[i], kernel_sizes[i], separable=separable)
    
    if not any(empty(cap)):
        if not is_scalar(cap):
            for i in range(1, len(cap)+1):
                stack[-i] = top = cap[i-1](top)
        else:
            stack[-1] = top = cap(top)

    return stack if return_all else top


# TODO: Start here! 
# TODO: consider this as dead code?  Convert to keras functional API and redo

# ' @export
conv_tower < - function(input,
                        filter_sizes=c(32, 64, 128, 256),
                        blocks=if (is_scalar(filter_sizes)) 4 else length(filter_sizes),
                        block_heights=3,  # depth of conv_blocK_1d
                        kernel_sizes=8,
                        block_caps=layer_max_pooling_1d,
                        last_cap=if (length(block_caps) == blocks)
                        last(block_caps)
                        else if (grepl("all", return))
                        layer_max_pooling_1d
                        else
                        layer_global_max_pooling_1d,
                        separable=TRUE,
                        return="top",
                        ...) {
    stopifnot(is_scalar_integerish(blocks), blocks >= 1L)
    force(last_cap)
    force(block_heights)

    if (is_scalar(block_heights))
    block_heights % < > % rep(blocks)
    if (is_scalar(kernel_sizes))
    kernel_sizes % < > % rep(blocks)
    if (is_scalar(filter_sizes))
    filter_sizes % < > % rep(blocks)

    if (length(block_caps) == 1) {
        block_caps < - lapply(1: blocks, function(x) block_caps)
        if (!is .list(last_cap))
        last_cap % < > % list()
        block_caps[blocks + 1 - rev(seq_along(last_cap))] < - last_cap
    } else
    stopifnot(length(block_caps) == blocks)

    stopifnot(return % in % c("top", "block_caps", "all_conv_layers", "all"))

    return_all < - return % in % c("all", "all_conv_layers")

    top < - input
    tower < - vector("list", blocks)

    for (i in seq_len(blocks)) {
        tower[[i]] < - top < -
        conv_block_1d(
            top,
            filter_sizes=filter_sizes[[i]],
            kernel_sizes=kernel_sizes[[i]],
            depth=block_heights[[i]],
            cap=block_caps[[i]],
            separable=separable,
            return_all=return_all,
            ...
        )
        # browser()
        # if (return_all)
        #   top <- top[[length(top)]]
        #
        # if (return == "all_conv_layers") # null out caps
        #   tower[[i]][ block_heights[[i]] + 1L ] <- NULL
    }

    switch(return,
           top=top,
           block_tops=tower,
           all_conv_layers=,
           all=unlist(tower, recursive=FALSE))
}
