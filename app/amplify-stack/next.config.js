// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
module.exports = {
  async redirects() {
    return [
      {
        source: '/',
        destination: '/authenticated',
        permanent: true,
      }
    ];
  },
  env: {
    USER_BRANCH: process.env.USER_BRANCH
  }
};
