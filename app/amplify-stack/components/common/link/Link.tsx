// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React from "react";
import NextLink, { LinkProps as NextLinkProps } from "next/link";
import styles from "./Link.module.scss";
import classNames from "classnames";

interface LinkProps extends NextLinkProps {
  className?: string;
  light?: boolean;
  underline?: boolean;
  download?: boolean;
}

export const Link: React.FC<LinkProps> = ({ children, className, light, underline, download, ...props }) => {
  return (
    <NextLink {...props}>
      <a
        className={classNames(
          className,
          styles.link,
          underline && styles["link--underlined"],
          light && styles["link--light"]
        )}
        download={download}
      >
        {children}
      </a>
    </NextLink>
  );
};
