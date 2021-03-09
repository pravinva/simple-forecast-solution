// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React from "react";
import styles from "./AdminSidebar.module.scss";
import { NavList } from "../nav-list/NavList";
import { NavListItem } from "../nav-list/item/NavListItem";
import { useRouter } from "next/router";
import { Link } from "../../../common/link/Link";

export const AdminSidebar: React.FC = () => {
  const router = useRouter();

  return (
    <div className={styles.body}>
      <NavList>
        <NavListItem active={router.asPath === "/authenticated/admin"}>
          <Link href={"/authenticated/admin"}>Admin Panel</Link>
        </NavListItem>
      </NavList>
    </div>
  );
};
