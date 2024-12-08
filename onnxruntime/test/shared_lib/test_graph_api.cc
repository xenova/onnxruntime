// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"

#include "onnxruntime_config.h"
#include "providers.h"
#include "test_allocator.h"
#include "test_fixture.h"
#include "utils.h"
#include "custom_op_utils.h"
#include <gsl/gsl>
